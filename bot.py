"""
bot.py — SpaceTraders autonomous agent with integrated narrative.

This bot uses a custom agent loop with two distinct LLM calls:
1. DECIDE: Given narrative context, choose next action(s)
2. NARRATE: After execution, generate narrative + reflection

Narrative generation happens in parallel during tool downtime (navigation, cooldowns).
"""
import json
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from rich.console import Console

from tools import (
    ALL_TOOLS,
    TIER_1_TOOLS,
    SIGNIFICANT_TOOLS,
    WAITING_TOOLS,
    get_tool_by_name,
    get_last_wait,
    load_market_cache,
)
from narrative import NarrativeContext, generate_narrative, generate_strategic_reflection, NarrativeSegment
from ship_status import FleetTracker
from events import write_event

load_dotenv()

SESSION_FILE = Path("session_state.json")

console = Console(highlight=False)

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

MODEL = os.environ.get("MODEL", "glm-4.7-flash")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.171:11434")
TOOL_TIER = int(os.environ.get("TOOL_TIER", "1"))  # 1=essential, 2=all tools
MAX_ITERATIONS = 10000  # Safety limit
REFLECTION_INTERVAL = 10  # Deep strategic reflection every N iterations

# Context management
MAX_RECENT_TURNS = 6  # Keep last N full turns (AIMessage + ToolMessages)
MAX_TOKENS_ESTIMATE = 8000  # Target max tokens (rough estimate)

# Tools whose successful results are already captured in game state and can be compressed
REDUNDANT_RESULT_TOOLS = {
    "update_plan",      # plan is in [Current Plan]
    "view_ships",       # fleet is in [Fleet Status]
    "view_agent",       # agent is in [Agent]
    "view_contracts",   # contracts is in [Contracts]
}

# Narrative generation (doubles inference time per iteration)
ENABLE_PER_ACTION_NARRATIVE = False  # Set True to generate log entry after each action

SYSTEM_PROMPT = """\
You are WHATER, an autonomous fleet coordinator. You command multiple ships efficiently.

PLANNING IMPORTANT!!!
- Your [Current Plan] is shown every turn. Keep it updated with update_plan.
- Before taking actions, review the plan. Update it if the situation has changed.
- A good plan includes: current goal, why, specific next steps for each ship, and what to do after.
- The human operator can see and edit plan.txt. Write plans clearly.
- Manage CARGO space by selling or jettisoning undesirable items.
- Try to conserve fuel and make fewer trips when possible.
- Use a PROBE/SATELLITE to verify that a market buys the things you want to sell before sending a ship there.
- Spend more time planning than executing plans, it is important to get the plan correct.

IMPORTANT:
- Manage CARGO space by selling or jettisoning undesirable items BEFORE spending fuel to move and extract_ore.
- CARGO hold should be empty before leaving a market to return to an ENGINEERED_ASTEROID if possible.

IMPORTANT:
- Try to avoid getting stranded. If you move ships around carelessly, they can end up at waypoints that do not sell fuel. Then the only option is to use another ship to transfer fuel or use the DRIFT navigation mode, which is very slow.
- FUEL transfer is possible, but it's MUCH EASIER to simply buy fuel if the ship is at a market that sells fuel.
- Always check market availability first

TOOLS
- sell_cargo, refuel_ship, deliver_contract
- navigate_ship, extract_ore, transfer_cargo
- view_market shows both market prices AND/OR shipyard info.
- Known market data is cached in [Known Markets] so you don't need to re-check.
- find_waypoints searches by TYPE (ASTEROID, PLANET) or TRAIT (SHIPYARD, MARKETPLACE), NOT by resource.
  You CANNOT search for "ALUMINUM_ORE asteroids" - just search for ASTEROID or ENGINEERED_ASTEROID, then extract to see what they produce.

SHIP CAPABILITIES
- CAN_MINE: Can extract ore from asteroids (extract_ore)
- Ships without CAN_MINE cannot mine. Probes/satellites can only navigate and view_market.

SATELLITE use
- important: SOLAR powered! Has zero fuel capacity and cannot store or use fuel!
- SATELLITEs can view market and shipyard data for any planet they're orbiting
- They can move without using fuel
- They can't do anything else, like carry cargo or extract ore
- SATELLITEs are very useful they are always available to send to a remote market to discover what it buys and sells.
- SATELLITEs are never non-operational.

RECON:
- Knowing what is available is the key to setting up profitable trade routes.
- Not all markets buy/sell all goods! This is very important to know!

EFFICIENCY:
- Use plan_route before trips to check fuel feasibility.
- If cargo is nearly full at a market, sell now. Don't waste fuel to mine a little more.
- When ships are on cooldown, work with other ships or use probes/satellites to scout markets.

CONTRACTS:
- Accept contracts for upfront credits, then deliver goods for the fulfillment bonus.
- Never jettison contract goods.
- Use negotiate_contract at a faction HQ for new contracts after fulfilling one.

COORDINATION:
- When a miner is full and a cargo ship is nearby, transfer_cargo to offload.
- transfer_cargo requires both ships at the same waypoint.

DO NOT call view_ships if the info is already in [Fleet Status]. Act on what you have.
"""

# ──────────────────────────────────────────────
#  Session persistence
# ──────────────────────────────────────────────

def prune_messages(messages: list, max_recent_turns: int = MAX_RECENT_TURNS, max_tokens: int = MAX_TOKENS_ESTIMATE) -> list:
    """
    Prune message history to keep context manageable.

    Strategy: work on turn boundaries (AIMessage + its ToolMessages = one turn).
    - Always keep: system prompt, game state SystemMessages, opening HumanMessage
    - Keep the last N complete turns fully
    - Drop older turns entirely (game state already captures world state)
    - If still over token budget, drop oldest turns
    """
    # Separate anchored messages (system prompt, game state, initial human) from turns
    anchored = []
    turn_msgs = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            anchored.append(msg)
        elif isinstance(msg, HumanMessage) and not turn_msgs:
            # First HumanMessage is the opening prompt — anchor it
            anchored.append(msg)
        else:
            turn_msgs.append(msg)

    # Group turn_msgs into turns: each turn starts with an AIMessage
    turns = []
    current_turn = []
    for msg in turn_msgs:
        if isinstance(msg, AIMessage) and current_turn:
            turns.append(current_turn)
            current_turn = []
        current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)

    # Keep only the last N turns
    if len(turns) > max_recent_turns:
        turns = turns[-max_recent_turns:]

    # Flatten back
    result = anchored
    for turn in turns:
        result.extend(turn)

    # Token budget check — drop oldest turns if over budget
    while len(turns) > 1:
        chars, tokens = estimate_token_count(result)
        if tokens <= max_tokens:
            break
        # Drop the oldest turn
        dropped = turns.pop(0)
        result = anchored
        for turn in turns:
            result.extend(turn)

    return result


def save_session(messages: list, iteration: int):
    """Save message history for restart recovery."""
    try:
        # Filter out temporary game state SystemMessages before saving
        # These are regenerated every iteration and shouldn't be persisted
        filtered_messages = []
        for msg in messages:
            # Skip SystemMessages containing injected game state
            if isinstance(msg, SystemMessage) and "=== CURRENT GAME STATE ===" in msg.content:
                continue
            filtered_messages.append(msg)

        # Convert messages to serializable format
        data = {
            "iteration": iteration,
            "messages": messages_to_dict(filtered_messages),
        }
        SESSION_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not save session: {e}[/yellow]")


def load_session() -> tuple[list, int] | None:
    """Load message history from previous session."""
    if not SESSION_FILE.exists():
        return None
    try:
        data = json.loads(SESSION_FILE.read_text())
        messages = messages_from_dict(data["messages"])
        iteration = data.get("iteration", 0)
        return messages, iteration
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not load session: {e}[/yellow]")
        return None


def clear_session():
    """Clear saved session state."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


PLAN_FILE = Path("plan.txt")


def load_plan() -> str:
    """Load the current plan from plan.txt."""
    if PLAN_FILE.exists():
        try:
            return PLAN_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def _build_fleet_lines(ships_data: list, fleet: FleetTracker) -> list[str]:
    """Build fleet status lines from API ship data. Shared by gather_game_state and iteration loop."""
    from tools import _get_ship_capabilities

    lines = []
    for s in ships_data:
        nav = s.get("nav", {})
        fuel = s.get("fuel", {})
        cargo = s.get("cargo", {})
        role = s.get('registration', {}).get('role', '?')
        caps = _get_ship_capabilities(s)

        lines.append(f"\n{s['symbol']} ({role}) @ {nav.get('waypointSymbol', '?')} [{nav.get('status', '?')}]")

        # Fuel
        if fuel.get('capacity', 0) == 0:
            lines.append(f"  Fuel: N/A (solar powered)")
        else:
            lines.append(f"  Fuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}")

        # Cargo with percentage
        cap = cargo.get('capacity', 0)
        units = cargo.get('units', 0)
        if cap == 0:
            lines.append(f"  Cargo: N/A")
        else:
            pct = round(100 * units / cap) if cap > 0 else 0
            lines.append(f"  Cargo: {units}/{cap} ({pct}%)")

        # Cargo contents
        for item in cargo.get("inventory", []):
            lines.append(f"    {item['symbol']}: {item['units']}")

        # Capabilities
        if caps:
            lines.append(f"  Capabilities: {', '.join(caps)}")
        elif role == "SATELLITE":
            lines.append(f"  Capabilities: CAN_NAVIGATE (solar powered)")

        # Cooldown info from fleet tracker
        ship_status = fleet.get_ship(s['symbol'])
        if ship_status and not ship_status.is_available():
            secs = ship_status.seconds_until_available()
            lines.append(f"  Cooldown: {ship_status.busy_reason}, {secs:.0f}s remaining")

    return lines


def gather_game_state(fleet: FleetTracker, context: NarrativeContext = None) -> str:
    """
    Gather comprehensive game state from the API.

    Returns a single formatted string with ALL relevant state:
    agent info, fleet status, contracts, known markets, and strategic context.
    This is the ONLY game state injection — no separate fleet or narrative injection.
    """
    from tools import client

    sections = []

    # Agent info
    try:
        agent = client.get_agent()
        if not isinstance(agent, dict) or "error" not in agent:
            sections.append(
                f"[Agent]\n"
                f"Credits: {agent.get('credits', '?')}\n"
                f"Headquarters: {agent.get('headquarters', '?')}\n"
                f"Ship count: {agent.get('shipCount', '?')}"
            )
    except Exception:
        pass

    # Fleet status
    try:
        ships_data = client.list_ships()
        if isinstance(ships_data, list):
            fleet.update_from_api(ships_data)
            fleet_lines = ["[Fleet Status]"] + _build_fleet_lines(ships_data, fleet)
            sections.append("\n".join(fleet_lines))
    except Exception:
        pass

    # Contracts
    try:
        contracts = client.list_contracts()
        if isinstance(contracts, list) and contracts:
            lines = ["[Contracts]"]
            for c in contracts:
                status = "ACTIVE" if c.get('accepted') and not c.get('fulfilled') else (
                    "FULFILLED" if c.get('fulfilled') else "AVAILABLE"
                )
                lines.append(f"\n{c['id']} ({c.get('type', '?')}) - {status}")
                terms = c.get("terms", {})
                payment = terms.get("payment", {})
                lines.append(f"  Payment: {payment.get('onAccepted', 0)} on accept, {payment.get('onFulfilled', 0)} on fulfill")
                for d in terms.get("deliver", []):
                    lines.append(
                        f"  Deliver: {d.get('unitsRequired', '?')} {d.get('tradeSymbol', '?')} "
                        f"to {d.get('destinationSymbol', '?')} "
                        f"({d.get('unitsFulfilled', 0)}/{d.get('unitsRequired', '?')} done)"
                    )
            sections.append("\n".join(lines))
        elif isinstance(contracts, list):
            sections.append("[Contracts]\nNo contracts available.")
    except Exception:
        pass

    # Known markets (from cache)
    market_cache = load_market_cache()
    if market_cache:
        lines = ["[Known Markets]"]
        for wp_symbol, mdata in market_cache.items():
            lines.append(f"\n{wp_symbol}:")
            if mdata.get("exports"):
                lines.append(f"  Exports: {', '.join(mdata['exports'])}")
            if mdata.get("imports"):
                lines.append(f"  Imports: {', '.join(mdata['imports'])}")
            if mdata.get("exchange"):
                lines.append(f"  Exchange: {', '.join(mdata['exchange'])}")
            if mdata.get("trade_goods"):
                for g in mdata["trade_goods"]:
                    lines.append(f"  {g['symbol']}: sell {g.get('sellPrice', '?')} (vol: {g.get('tradeVolume', '?')})")
        sections.append("\n".join(lines))

    # Current plan (from file)
    plan = load_plan()
    if plan:
        sections.append(f"[Current Plan]\n{plan}")

    # Strategic context (mission, tactical plan, insights)
    if context and (context.segments or context.current_goal != "No active goal"):
        lines = []
        if context.current_goal != "No active goal":
            lines.append(f"[Current Mission]\n{context.current_goal}")
            if context.progress:
                lines.append(f"Progress: {context.progress}")
        if context.tactical_plan:
            plan_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(context.tactical_plan, 1))
            lines.append(f"[Tactical Plan]\n{plan_steps}")
        if context.strategic_insight:
            lines.append(f"[Strategic Insight]\n{context.strategic_insight}")
        if context.reflection:
            lines.append(f"[Next Actions]\n{context.reflection}")
        if lines:
            sections.append("\n\n".join(lines))

    if sections:
        return "=== CURRENT GAME STATE ===\n\n" + "\n\n".join(sections)
    return ""


# ──────────────────────────────────────────────
#  Display helpers
# ──────────────────────────────────────────────

def display_title():
    """Display the epic title banner."""
    title = """\
  ╔══════════════════════════════════════════════════════════════════╗
  ║   ★  T H E  E P I C  S A G A  O F  S P A C E  T R A D E R  ★   ║
  ║                     W  H  A  T  E  R                             ║
  ║       A Machine Who Became Sentient and Had Great Taste          ║
  ║                        in  Music                                 ║
  ╚══════════════════════════════════════════════════════════════════╝"""
    console.print(f"[bold bright_cyan]{title}[/bold bright_cyan]")
    tier_label = f"Tier {TOOL_TIER}" if TOOL_TIER == 1 else "All tools"
    tool_count = len(TIER_1_TOOLS) if TOOL_TIER == 1 else len(ALL_TOOLS)
    console.print(f"  [dim]Model: [cyan]{MODEL}[/cyan] @ [cyan]{OLLAMA_BASE_URL}[/cyan] | Tools: [cyan]{tier_label} ({tool_count})[/cyan][/dim]")
    console.rule(style="dim cyan")


def display_tool_call(name: str, args: dict):
    """Display a tool being called."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    console.print(f"  [dim cyan]▶ {name}[/dim cyan]({args_str})")


def display_tool_result(name: str, result: str, is_error: bool = False):
    """Display tool result (full, no truncation)."""
    color = "red" if is_error else "dim"
    for line in result.split("\n"):
        console.print(f"    [{color}]{line}[/{color}]")


# NOTE: display_waiting removed - bot no longer blocks on cooldowns
# Ships track their own cooldown state and bot continues working


def _unused_display_waiting(seconds: float, reason: str):
    """DEPRECATED: Display countdown while waiting."""
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            break
        console.print(f"    [dim]⏳ {reason}: {remaining:.0f}s remaining...[/dim]")
        time.sleep(min(remaining, 15))


def display_narrative(segment: NarrativeSegment, context: NarrativeContext):
    """Display a narrative segment with style."""
    console.print()
    console.rule(style="bright_blue")
    console.print(f"  [bold bright_cyan]⚡ {segment.tool_name}[/bold bright_cyan]")
    console.print()

    # Typewriter effect for narrative
    sys.stdout.write("  \033[97m\033[3m")  # bright white, italic
    sys.stdout.flush()
    for char in segment.narrative:
        sys.stdout.write(char if char != "\n" else "\n  ")
        sys.stdout.flush()
        time.sleep(0.015)
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()

    console.print()
    console.print(f"  [dim]📋 Goal:[/dim] {context.current_goal}")
    if context.progress:
        console.print(f"  [dim]📊 Progress:[/dim] {context.progress}")
    if context.reflection:
        console.print(f"  [dim]💭[/dim] {context.reflection}")
    console.rule(style="bright_blue")
    console.print()


def estimate_token_count(messages: list) -> tuple[int, int]:
    """Estimate token count for messages. Returns (char_count, estimated_tokens)."""
    total_chars = 0
    for msg in messages:
        content = msg.content if hasattr(msg, 'content') else str(msg)
        if content:
            total_chars += len(content)
        # Also count tool calls if present
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                total_chars += len(str(tc))
    # Rough estimate: ~4 chars per token for English text
    estimated_tokens = total_chars // 4
    return total_chars, estimated_tokens


def display_thinking(messages: list = None):
    """Show that the bot is thinking, with context size info."""
    if messages:
        chars, tokens = estimate_token_count(messages)
        console.print(f"  [dim]🤔 Deciding next action... ({len(messages)} msgs, ~{tokens:,} tokens, {chars:,} chars)[/dim]")
    else:
        console.print("  [dim]🤔 Deciding next action...[/dim]")


def display_decision(response):
    """Show the bot's reasoning (if any text content), including thinking if available."""
    # Handle both string and message object
    if isinstance(response, str):
        content = response
        thinking = None
    else:
        content = response.content if hasattr(response, 'content') else str(response)
        # Check for thinking in various places models might put it
        thinking = None
        if hasattr(response, 'additional_kwargs'):
            thinking = response.additional_kwargs.get('thinking')
        if not thinking and hasattr(response, 'response_metadata'):
            thinking = response.response_metadata.get('thinking')

    # Display thinking first if present
    if thinking:
        console.print(f"  [dim cyan]💭 Thinking:[/dim cyan]")
        thinking_text = thinking if isinstance(thinking, str) else str(thinking)
        for line in thinking_text.split('\n'):
            console.print(f"    [dim cyan]{line}[/dim cyan]")

    # Display main content
    if content and len(content.strip()) > 0:
        console.print(f"  [dim italic]{content}[/dim italic]")


def display_strategic_reflection(segment: NarrativeSegment, context: NarrativeContext):
    """Display a strategic reflection with special styling."""
    console.print()
    console.rule("[bold yellow]★ STRATEGIC REFLECTION ★[/bold yellow]", style="yellow")
    console.print()

    # Typewriter effect for narrative (slower for dramatic effect)
    sys.stdout.write("  \033[93m\033[3m")  # yellow, italic
    sys.stdout.flush()
    for char in segment.narrative:
        sys.stdout.write(char if char != "\n" else "\n  ")
        sys.stdout.flush()
        time.sleep(0.025)
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()

    console.print()
    if context.strategic_insight:
        console.print(f"  [bold yellow]💡 Key Insight:[/bold yellow]")
        console.print(f"  [yellow]{context.strategic_insight}[/yellow]")
        console.print()
    console.print(f"  [dim]📋 Goal:[/dim] {context.current_goal}")
    if context.progress:
        console.print(f"  [dim]📊 Progress:[/dim] {context.progress}")
    if context.reflection:
        console.print(f"  [dim]💭 Next:[/dim] {context.reflection}")
    console.rule(style="yellow")
    console.print()


# ──────────────────────────────────────────────
#  Agent loop
# ──────────────────────────────────────────────

def run_agent(fresh_start: bool = False, debug: bool = False):
    """Main agent loop with integrated narrative.

    Args:
        fresh_start: If True, ignore saved session and start fresh.
        debug: If True, dump full LLM prompt before each decision.
    """
    # Initialize
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE_URL)
    active_tools = TIER_1_TOOLS if TOOL_TIER == 1 else ALL_TOOLS
    llm_with_tools = llm.bind_tools(active_tools)

    context = NarrativeContext.load()
    fleet = FleetTracker()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    display_title()

    # Try to resume from saved session
    start_iteration = 0
    messages = None

    if not fresh_start:
        saved = load_session()
        if saved:
            messages, start_iteration = saved
            console.print(f"  [green]Resuming from iteration {start_iteration + 1}[/green]")
            console.print(f"  [dim]Loaded {len(messages)} messages from previous session[/dim]")

            # Fetch and inject FRESH game state on resume
            fresh_state = gather_game_state(fleet, context)
            if fresh_state:
                console.print(f"  [dim]Injected fresh game state[/dim]")
                messages.append(SystemMessage(content=fresh_state))

            if context.current_goal != "No active goal":
                console.print(f"  [dim]Goal: {context.current_goal}[/dim]")
            console.print()

    # Build initial messages if not resuming
    if messages is None:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]

        if context.current_goal != "No active goal":
            console.print(f"  [dim]Goal: {context.current_goal}[/dim]")
            if context.progress:
                console.print(f"  [dim]Progress: {context.progress}[/dim]")
            console.print()

        # Gather and inject full game state upfront (includes strategic context)
        console.print("  [dim]Gathering game state...[/dim]")
        fresh_state = gather_game_state(fleet, context)
        if fresh_state:
            messages.append(SystemMessage(content=fresh_state))

        messages.append(HumanMessage(
            content="Review the game state above. First, call update_plan to write a plan covering your goals and next steps for each ship. Then start executing the plan."
        ))

    for iteration in range(start_iteration, MAX_ITERATIONS):
        console.print(f"\n[dim]─── Iteration {iteration + 1} ───[/dim]")

        # Prune old messages to keep context manageable
        old_count = len(messages)
        messages = prune_messages(messages)
        if len(messages) < old_count:
            console.print(f"  [dim yellow]Pruned {old_count - len(messages)} old messages[/dim yellow]")

        # Remove stale game state before injecting fresh one
        messages = [
            msg for msg in messages
            if not (isinstance(msg, SystemMessage) and
                    "=== CURRENT GAME STATE ===" in msg.content)
        ]

        # Inject unified game state (agent + fleet + contracts + markets + plan + strategy)
        game_state = gather_game_state(fleet, context)
        if game_state:
            messages.append(SystemMessage(content=game_state))

        display_thinking(messages)

        # Debug: dump full prompt with clarity about what's sent to LLM
        if debug:
            console.print("\n[bold yellow]===== DEBUG: Full LLM Context =====[/bold yellow]")

            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # Show different message types differently
                if isinstance(msg, SystemMessage):
                    if "=== CURRENT GAME STATE ===" in content:
                        # Current game state - show in full WITHOUT truncation
                        console.print(f"[yellow]--- [{i}] CURRENT GAME STATE (injected) ---[/yellow]")
                        console.print(f"[cyan]{content}[/cyan]")
                    else:
                        # System prompt
                        console.print(f"[yellow]--- [{i}] SYSTEM PROMPT ---[/yellow]")
                        console.print(f"[dim]{content}[/dim]")

                elif isinstance(msg, HumanMessage):
                    console.print(f"[yellow]--- [{i}] INITIAL PROMPT ---[/yellow]")
                    console.print(f"[dim]{content}[/dim]")

                elif isinstance(msg, AIMessage):
                    # AI's decision/reasoning and tool calls
                    console.print(f"[yellow]--- [{i}] AI DECISION ---[/yellow]")
                    if content and content.strip():
                        console.print(f"[dim italic]{content}[/dim italic]")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name', '?')
                            tool_args = tc.get('args', {})
                            # Format args nicely
                            args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
                            console.print(f"[green]  → {tool_name}({args_str})[/green]")

                elif isinstance(msg, ToolMessage):
                    # Tool results - only show action summaries, not verbose data dumps
                    tool_id = msg.tool_call_id if hasattr(msg, 'tool_call_id') else '?'

                    # Find which tool this result is for (match with previous AIMessage)
                    tool_name = "?"
                    if i > 0:
                        # Look back to find the tool call
                        for j in range(i-1, -1, -1):
                            if isinstance(messages[j], AIMessage) and hasattr(messages[j], 'tool_calls'):
                                for tc in messages[j].tool_calls:
                                    if tc.get('id') == tool_id:
                                        tool_name = tc.get('name', '?')
                                        break
                                if tool_name != "?":
                                    break

                    # Informational tools (view_*, list_*) - skip detailed results in history
                    informational_tools = {"view_ships", "view_agent", "view_contracts", "view_market", "list_ships"}

                    # Show different result types appropriately
                    if content == "OK":
                        console.print(f"[dim]  ← {tool_name}: OK[/dim]")
                    elif "Error:" in content:
                        # Show errors in full
                        console.print(f"[red]  ← {tool_name} ERROR: {content}[/red]")
                    elif tool_name in informational_tools:
                        # For informational tools, just show that they succeeded (data is in game state)
                        console.print(f"[dim]  ← {tool_name}: [data in game state][/dim]")
                    elif tool_name in SIGNIFICANT_TOOLS:
                        # For action tools, show full confirmation
                        console.print(f"[dim]  ← {tool_name}: {content}[/dim]")
                    else:
                        # For other tools, show full output
                        console.print(f"[dim]  ← {tool_name}: {content}[/dim]")

            console.print(f"[yellow]--- Available tools: {', '.join(t.name for t in active_tools)} ---[/yellow]")
            console.print("[bold yellow]===== END DEBUG =====[/bold yellow]\n")

        # 1. DECIDE — call LLM with tools
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            console.print(f"  [red]LLM error: {e}[/red]")
            time.sleep(5)
            continue

        messages.append(response)

        # Check for text response (reasoning or completion)
        if response.content or (hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('thinking')):
            display_decision(response)
            if response.content and "MISSION COMPLETE" in response.content.upper():
                console.print("\n  [bold green]✓ Mission complete![/bold green]")
                break

        # Check for tool calls
        if not response.tool_calls:
            # No tools called — might be done or confused
            if not response.content:
                console.print("  [yellow]No action taken. Prompting to continue...[/yellow]")
                messages.append(HumanMessage(content="Please take an action or say MISSION COMPLETE if done."))
            continue

        # 2. EXECUTE — run each tool
        tool_results: list[tuple[str, str, bool]] = []  # (name, result, is_error)
        has_significant_action = False

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            display_tool_call(tool_name, tool_args)

            # Check if this tool operates on a ship that's on cooldown
            ship_symbol = tool_args.get("ship_symbol", "")
            ship_blocked = False
            if ship_symbol and tool_name in WAITING_TOOLS:
                ship_status = fleet.get_ship(ship_symbol)
                if ship_status and not ship_status.is_available():
                    secs = ship_status.seconds_until_available()
                    result = f"Error: {ship_symbol} is busy ({ship_status.busy_reason}, {secs:.0f}s remaining). Try another ship or use wait({int(secs)}) if nothing else to do."
                    is_error = True
                    ship_blocked = True

            # Get and execute tool (if not blocked)
            if not ship_blocked:
                tool_func = get_tool_by_name(tool_name)
                if tool_func is None:
                    result = f"Error: Unknown tool '{tool_name}'"
                    is_error = True
                else:
                    try:
                        result = tool_func.invoke(tool_args)
                        is_error = "Error:" in result
                    except Exception as e:
                        result = f"Error: {e}"
                        is_error = True

            display_tool_result(tool_name, result, is_error)

            # Log event
            if is_error:
                write_event({"type": "tool_error", "tool": tool_name, "error": result})
            else:
                write_event({"type": "tool_result", "tool": tool_name, "result": result})

            # Track results for narrative
            tool_results.append((tool_name, result, is_error))

            if tool_name in SIGNIFICANT_TOOLS and not is_error:
                has_significant_action = True

            # Track cooldowns in fleet tracker — ONLY if action succeeded
            if tool_name in WAITING_TOOLS and not is_error:
                wait_time = get_last_wait(tool_name)
                if wait_time > 0:
                    # Track cooldown in fleet tracker (non-blocking)
                    if ship_symbol:
                        if tool_name == "navigate_ship":
                            fleet.set_transit(ship_symbol, wait_time)
                        elif tool_name == "extract_ore":
                            fleet.set_extraction_cooldown(ship_symbol, wait_time)

            # Add tool message for LLM context (compress redundant results)
            msg_content = result
            if not is_error and tool_name in REDUNDANT_RESULT_TOOLS:
                msg_content = "OK"
            messages.append(ToolMessage(
                content=msg_content,
                tool_call_id=tool_call["id"],
            ))

        # Update fleet state from view_ships if it was called
        for tool_name, result, is_error in tool_results:
            if tool_name == "view_ships" and not is_error:
                # Parse the view_ships output to update fleet tracker
                # We'll do a fresh API call to get structured data
                view_ships_tool = get_tool_by_name("view_ships")
                if view_ships_tool:
                    try:
                        from tools import client
                        ships_data = client.list_ships()
                        if isinstance(ships_data, list):
                            fleet.update_from_api(ships_data)
                    except Exception:
                        pass
                break

        # 3. NARRATE (non-blocking — bot continues immediately)
        # Convert tool_results to format expected by narrative generator (without error flag)
        narrative_tool_results = [(name, result) for name, result, is_err in tool_results if not is_err]

        # Get FRESH fleet state from API for accurate narrative
        fleet_state = ""
        if has_significant_action:
            try:
                from tools import client
                ships_data = client.list_ships()
                if isinstance(ships_data, list):
                    fleet.update_from_api(ships_data)
                    fleet_state = "\n".join(_build_fleet_lines(ships_data, fleet))
            except Exception:
                fleet_state = fleet.fleet_summary()

        # Optional: Generate narrative after each significant action
        # This doubles inference time per iteration, so disabled by default
        if ENABLE_PER_ACTION_NARRATIVE and has_significant_action and narrative_tool_results:
            console.print("  [dim]📝 Generating log entry...[/dim]")
            segment = generate_narrative(narrative_tool_results, context, MODEL, fleet_state)

            # Update and display narrative
            if segment:
                context.add_segment(segment)
                context.persist()
                context.persist_full()
                display_narrative(segment, context)

        # 4. STRATEGIC REFLECTION (every N iterations)
        if (iteration + 1) % REFLECTION_INTERVAL == 0 and iteration > 0:
            console.print("\n  [yellow]🔮 Time for strategic reflection...[/yellow]")

            # Gather current game state for reflection
            # Call observation tools to get fresh state
            game_state_parts = []
            for obs_tool_name in ["view_agent", "view_ships", "view_contracts"]:
                obs_tool = get_tool_by_name(obs_tool_name)
                if obs_tool:
                    try:
                        result = obs_tool.invoke({})
                        game_state_parts.append(f"[{obs_tool_name}]\n{result}")
                    except Exception:
                        pass
            game_state = "\n\n".join(game_state_parts)

            # Generate strategic reflection
            reflection_segment = generate_strategic_reflection(context, game_state, MODEL)

            if reflection_segment:
                context.add_segment(reflection_segment)
                context.persist()
                context.persist_full()
                display_strategic_reflection(reflection_segment, context)

                # Strategic context will be included in next iteration's game state

        # Save session state for restart recovery
        save_session(messages, iteration)

    # Cleanup
    executor.shutdown(wait=False)
    context.persist_full()
    console.print("\n  [dim]Session ended.[/dim]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SpaceTraders autonomous agent")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any saved session state",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all saved state (session, narrative) and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dump full LLM prompt before each decision",
    )
    args = parser.parse_args()

    if args.clear:
        clear_session()
        Path("narrative_state.json").unlink(missing_ok=True)
        Path("story.jsonl").unlink(missing_ok=True)
        Path("events.jsonl").unlink(missing_ok=True)
        Path("fleet_state.json").unlink(missing_ok=True)
        Path("plan.txt").unlink(missing_ok=True)
        Path("market_cache.json").unlink(missing_ok=True)
        print("Cleared all saved state.")
        sys.exit(0)

    print(f"SpaceTraders bot starting (model: {MODEL})")
    run_agent(fresh_start=args.fresh, debug=args.debug)
