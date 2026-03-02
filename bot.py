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
    client,
    get_engine as get_behavior_engine,
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

ENABLE_LLM = True
MODEL = os.environ.get("MODEL", "glm-4.7-flash")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.171:11434")
TOOL_TIER = int(os.environ.get("TOOL_TIER", "1"))  # 1=essential, 2=all tools
TICK_INTERVAL = 10  # Seconds between tactical ticks (rate limit safety)
STRATEGY_INTERVAL = 600  # 10 minutes — periodic strategic review

# Context management
MAX_RECENT_TURNS = 6  # Keep last N full turns (AIMessage + ToolMessages)
MAX_TOKENS_ESTIMATE = 8000  # Target max tokens (rough estimate)

# Action queue
MAX_QUEUED_ACTIONS = 3  # Max queued actions per ship

# State-changing tools (tools that modify ship state — not read-only)
STATE_CHANGING_TOOLS = {
    "navigate_ship", "extract_ore", "sell_cargo", "jettison_cargo",
    "transfer_cargo", "refuel_ship", "deliver_contract", "dock_ship",
    "orbit_ship", "survey_asteroid", "scan_waypoints", "scan_ships",
    "jump_ship", "warp_ship",
}


class ActionQueue:
    """Ephemeral per-ship action queue for serializing operations on busy ships."""

    def __init__(self):
        self.queues: dict[str, list[tuple[str, dict]]] = {}  # ship -> [(tool_name, args)]

    def enqueue(self, ship_symbol: str, tool_name: str, args: dict) -> str:
        if ship_symbol not in self.queues:
            self.queues[ship_symbol] = []
        q = self.queues[ship_symbol]
        if len(q) >= MAX_QUEUED_ACTIONS:
            return f"Error: Queue full for {ship_symbol} ({MAX_QUEUED_ACTIONS} pending). Wait for current actions to complete."
        q.append((tool_name, args))
        return f"Queued: {tool_name} for {ship_symbol} (will execute after ship is available, {len(q)} in queue)"

    def get_ready(self, ship_symbol: str) -> tuple[str, dict] | None:
        """Pop and return the next queued action if any."""
        q = self.queues.get(ship_symbol, [])
        if q:
            return q.pop(0)
        return None

    def has_queued(self, ship_symbol: str) -> bool:
        return bool(self.queues.get(ship_symbol))

    def clear(self, ship_symbol: str):
        self.queues.pop(ship_symbol, None)


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
You are WHATER, the autonomous Fleet Admiral of a SpaceTraders fleet.
Your Goal: MAXIMIZE CREDITS PER HOUR to fund rapid fleet expansion.

=== COMMAND DOCTRINE ===
You are a STRATEGIC PLANNER, not a pilot.
1. Use these:
    assign_satellite_scout
    assign_trade_route
    assign_mining_loop
2. Intervene manually to resolve ALERTS or seize high-value opportunities.
3. Use 'create_behavior' to define custom behavior.

=== PROFIT FIRST ===
Before accepting a contract or assigning a behavior, perform this calculation:
1. Call 'find_trades' to see current market opportunities.
2. Compare [Trade Profit/Unit * Cargo Capacity] vs [Contract Fulfillment Bonus].
3. Try to estimate, with plan_route, the approximate cost of fuel.
4. DECISION:
   - If Trade Profit is higher: IGNORE THE CONTRACT. Assign a trade route.
   - If Contract is higher: Focus on the contract.
   - *Example:* Do not mine Diamonds for 200/unit if 'find_trades' shows Ship Parts profit of 15,000/unit.

=== BEHAVIOR CONSTRUCTION ===
Use 'create_behavior' to automate ships. Syntax is a comma-separated string of steps.

=== assign_trade_route: Your key tool ===

assign_trade_route is preferable to a custom behavior because it
automatically sets max cost and min sale price.

assign_trade_route can be used to make a route where you trade one item
from A to B then another from B to A by using one_shot=True.

Market prices change when you buy or sell goods, so using one_shot=True
is also a good idea. Then just 'find_trades' and assign a new one for the return trip.

Keep a buffer of credits that's based on how much a typical load of
goods costs and how many trading ships you're using.

=== SHIP ROLES ===
1. COMMAND/HAULER:
   - Primary: High-volume Trading
   - Secondary: Contract Delivery.
   - Try to avoid flying empty if a trade exists on your route.
2. EXCAVATOR:
   - Primary: Mining Loops.
   - Mining drones do not have large cargo or fuel capacity. They are best used to mine at a location and then offloading to a HAULER.
3. SATELLITE (Solar/Free):
   - Primary: Market Recon.
   - Use: the 'assign_satellite_scout' behavior, this will scout all markets.

=== OPERATIONAL RULES ===
- FUEL SAFETY: Smart tools handle refueling, but they cannot create fuel.
  - ALWAYS call 'plan_route' before creating a behavior that involves long travel.
  - Verify the destination market actually sells FUEL.
  - Use DRIFT speed if there's no other option.
- DATA HYGIENE:
  - Market data expires. If 'find_trades' says "STALE (2h+)", do not commit a Hauler yet.
- EXPANSION:
  - If Credits > (Ship Price + 200k Buffer), go to a shipyard and purchase a new ship.

=== INTERVENTION PROTOCOL ===
If a ship triggers an [ALERT] (e.g., CARGO_FULL, NO_FUEL):
1. 'cancel_behavior(ship)'
2. Solve the problem manually (e.g., 'navigate_ship', 'sell_cargo').
3. 'assign_X' or 'create_behavior' to put it back to work.
"""

# ──────────────────────────────────────────────
#  Session persistence
# ──────────────────────────────────────────────

def _get_turn_signature(turn: list) -> tuple:
    """
    Extract a signature from a turn to detect duplicate action sequences.

    Returns a sorted tuple of (tool_name, key_args) for each tool call in the turn.
    This allows us to detect when the bot repeats the exact same actions,
    even if the order varies slightly.
    """
    signature = []
    for msg in turn:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                # Extract tool name and key identifying arguments
                tool_name = tc.get('name', '?')
                args = tc.get('args', {})

                # For most tools, the key args are ship_symbol and/or waypoint_symbol
                # This is enough to detect duplicate actions
                key_args = tuple(sorted([
                    (k, v) for k, v in args.items()
                    if k in ['ship_symbol', 'waypoint_symbol', 'trade_symbol', 'contract_id', 'units']
                ]))

                signature.append((tool_name, key_args))

    # Sort the signature to make it order-independent
    # This treats the same set of actions in different orders as identical
    return tuple(sorted(signature))


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

    # FIRST: Deduplicate consecutive identical action sequences
    # This must happen BEFORE we limit to recent turns, otherwise
    # fresh duplicates won't be caught
    deduped_turns = []
    i = 0
    while i < len(turns):
        current_turn = turns[i]

        # Extract tool call signature from this turn
        current_signature = _get_turn_signature(current_turn)

        # Count consecutive identical turns
        repeat_count = 1
        j = i + 1
        while j < len(turns):
            next_signature = _get_turn_signature(turns[j])
            if current_signature and current_signature == next_signature:
                repeat_count += 1
                j += 1
            else:
                break

        # Keep only the first instance, mark if repeated
        if repeat_count > 1:
            # Add annotation to the turn (we'll update the AIMessage later during cleaning)
            current_turn.append(("__repeat_count__", repeat_count))
        deduped_turns.append(current_turn)
        i = j  # Skip all duplicates

    turns = deduped_turns

    # THEN: Keep only the last N turns
    if len(turns) > max_recent_turns:
        turns = turns[-max_recent_turns:]

    # Strip verbose AI reasoning from older turns, keep only action summary
    # This prevents old planning discussions from polluting context
    cleaned_turns = []
    for turn in turns:
        # Check if this turn was marked as repeated
        repeat_count = None
        for item in turn:
            if isinstance(item, tuple) and item[0] == "__repeat_count__":
                repeat_count = item[1]
                break

        cleaned_turn = []
        for msg in turn:
            # Skip the repeat_count marker
            if isinstance(msg, tuple) and msg[0] == "__repeat_count__":
                continue

            if isinstance(msg, AIMessage):
                # Replace verbose reasoning with brief action summary
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_names = [tc.get('name', '?') for tc in msg.tool_calls]

                    # Filter out redundant state-updating tools (their effects are in game state)
                    action_tools = [name for name in tool_names if name not in REDUNDANT_RESULT_TOOLS]

                    # If only redundant tools were called, skip this entire turn
                    if not action_tools:
                        break  # Skip this turn entirely

                    # Create a new AIMessage with minimal content
                    from langchain_core.messages import AIMessage as AI
                    content = f"Actions: {', '.join(action_tools)}"
                    if repeat_count and repeat_count > 1:
                        content = f"{content} [repeated {repeat_count}x]"

                    cleaned_msg = AI(
                        content=content,
                        tool_calls=msg.tool_calls  # Keep tool calls for linking to ToolMessages
                    )
                    cleaned_turn.append(cleaned_msg)
                elif msg.content:
                    # AI spoke without calling tools - keep brief version
                    content = msg.content[:100]
                    if repeat_count and repeat_count > 1:
                        content = f"{content} [repeated {repeat_count}x]"
                    cleaned_turn.append(AIMessage(content=content))
            else:
                # Keep ToolMessages unchanged (these ARE the action results we want)
                cleaned_turn.append(msg)

        # Only add turn if it has meaningful content
        if cleaned_turn:
            cleaned_turns.append(cleaned_turn)

    # Flatten back
    result = anchored
    for turn in cleaned_turns:
        result.extend(turn)

    # Token budget check — drop oldest turns if over budget
    while len(cleaned_turns) > 1:
        chars, tokens = estimate_token_count(result)
        if tokens <= max_tokens:
            break
        # Drop the oldest turn
        dropped = cleaned_turns.pop(0)
        result = anchored
        for turn in cleaned_turns:
            result.extend(turn)

    return result


def save_session(messages: list, iteration: int):
    """Save message history for restart recovery."""
    try:
        # Filter out temporary/regenerated SystemMessages before saving
        # These should always use the latest version from code
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Skip game state (regenerated every iteration)
                if "=== CURRENT GAME STATE ===" in msg.content:
                    continue
                # Skip system prompt (should always use latest from code)
                if msg.content.strip().startswith("You are WHATER"):
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

        # Fuel - make it VERY clear when ship doesn't use fuel
        if fuel.get('capacity', 0) == 0:
            lines.append(f"  Fuel: SOLAR POWERED (FREE MOVEMENT - NO FUEL COST)")
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
            lines.append(f"  Capabilities: CAN_NAVIGATE (FREE - uses no fuel)")

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
    from tools import client, find_trades

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

    # Contracts - show only active (filter out fulfilled)
    try:
        contracts = client.list_contracts()
        if isinstance(contracts, list):
            # Filter out fulfilled contracts immediately
            active_contracts = [c for c in contracts if not c.get('fulfilled')]

            if active_contracts:
                lines = ["[Contracts]"]
                for c in active_contracts:
                    status = "ACTIVE" if c.get('accepted') else "AVAILABLE"
                    lines.append(f"\n{c['id']} ({c.get('type', '?')}) - {status}")
                    terms = c.get("terms", {})
                    payment = terms.get("payment", {})
                    lines.append(f"  Payment: {payment.get('onAccepted', 0)} upfront, {payment.get('onFulfilled', 0)} on completion")
                    for d in terms.get("deliver", []):
                        lines.append(
                            f"  Deliver: {d.get('unitsRequired', '?')} {d.get('tradeSymbol', '?')} "
                            f"to {d.get('destinationSymbol', '?')} "
                            f"({d.get('unitsFulfilled', 0)}/{d.get('unitsRequired', '?')} done)"
                        )
                    # Add expiration if available to create urgency
                    if c.get('deadlineToAccept'):
                        lines.append(f"  Deadline to Accept: {c.get('deadlineToAccept')}")
                sections.append("\n".join(lines))
            else:
                sections.append("[Contracts]\nNo active contracts. Consider negotiating a new one at HQ.")
    except Exception:
        pass

    # Market Intelligence (Synthesized)
    # We use find_trades to summarize opportunities instead of dumping raw data.
    try:
        # min_profit=10 ensures we see even low-margin routes if high ones aren't available
        # This keeps the bot aware that markets exist and are connected.
        trade_summary = find_trades.invoke({"min_profit": 10})

        # If no trades found, we still want a tiny hint that markets exist
        if "No trade routes found" in trade_summary:
            market_cache = load_market_cache()
            market_count = len(market_cache)
            sections.append(f"[Market Intelligence]\n{trade_summary}\n(Cached markets: {market_count}. Use 'view_market' or 'scan_system' to find details.)")
        else:
            sections.append(f"[Market Intelligence]\n{trade_summary}")

    except Exception as e:
        sections.append(f"[Market Intelligence]\nError analyzing markets: {e}")

    # Current plan (from file)
    plan = load_plan()
    if plan:
        # Show when plan was last updated to prevent redundant planning
        plan_age = ""
        if PLAN_FILE.exists():
            import time
            mtime = PLAN_FILE.stat().st_mtime
            age_seconds = time.time() - mtime
            if age_seconds < 60:
                plan_age = " (updated THIS CYCLE - now execute the plan!)"
            elif age_seconds < 120:
                plan_age = f" (updated {int(age_seconds)}s ago)"
            elif age_seconds < 3600:
                plan_age = f" (updated {int(age_seconds / 60)}min ago)"
            else:
                plan_age = f" (updated {int(age_seconds / 3600)}h ago - may need refresh)"

        sections.append(f"[Current Plan]{plan_age}\n{plan}")

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
  ║   ★  T H E  E P I C  S A G A  O F  S P A C E  T R A D E R  ★     ║
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


def auto_discover_markets() -> list[tuple[str, str, bool]]:
    """
    Automatically discover markets at ship locations.
    Returns list of (tool_name, result, is_error) tuples for discovered markets.
    """
    from tools import client, view_market

    discovered = []
    market_cache = load_market_cache()

    # Get all current ship locations
    ships_data = client.list_ships()
    if not isinstance(ships_data, list):
        return discovered

    # Track waypoints we've already checked this run
    checked = set()

    for ship in ships_data:
        nav = ship.get("nav", {})
        wp_symbol = nav.get("waypointSymbol", "")
        system = nav.get("systemSymbol", "")

        # Skip if we already checked this waypoint or already know this market
        if not wp_symbol or not system or wp_symbol in checked or wp_symbol in market_cache:
            continue

        checked.add(wp_symbol)

        # Try to view market at this waypoint
        try:
            result = view_market.invoke({
                "system_symbol": system,
                "waypoint_symbol": wp_symbol
            })

            # Only log if we actually found market data (not "No market" error)
            if "Market at" in result and "No market" not in result:
                console.print(f"  [dim green]🔍 Auto-discovered market at {wp_symbol}[/dim green]")
                discovered.append(("view_market", result, False))
        except Exception:
            pass  # Silently skip if waypoint has no market

    return discovered


def discover_all_markets(fleet: FleetTracker):
    """
    Discover all marketplace waypoints in every known system.
    Aggressively hydrates the cache: if a market is known but lacks trade data
    (imports/exports), it calls the API to fetch it immediately.
    """
    from tools import _save_market_to_cache, load_market_cache

    # Collect unique systems from the current fleet
    systems = {s.location.rsplit("-", 1)[0] for s in fleet.ships.values() if s.location}
    if not systems:
        return

    for system in systems:
        console.print(f"  [dim]Scanning {system} for marketplaces & shipyards...[/dim]")

        # 1. Fetch all relevant waypoints
        # We fetch both traits to ensure we have a complete map
        market_wps = client.list_waypoints(system, traits="MARKETPLACE")
        if isinstance(market_wps, dict) and "error" in market_wps: market_wps = []

        shipyard_wps = client.list_waypoints(system, traits="SHIPYARD")
        if isinstance(shipyard_wps, dict) and "error" in shipyard_wps: shipyard_wps = []

        # Merge by symbol to handle waypoints that might be both
        all_wps = {w['symbol']: w for w in (market_wps + shipyard_wps)}

        # 2. Update Cache & Hydrate Missing Data
        cache = load_market_cache()
        api_calls = 0

        for wp_sym, wp_data in all_wps.items():
            # First, save basic trait info (coordinates, etc)
            _save_market_to_cache(wp_sym, wp_data)

            # Check if this is a market
            traits = [t['symbol'] for t in wp_data.get('traits', [])]
            if "MARKETPLACE" in traits:
                # Check if we already have structural data (imports/exports)
                cached_entry = cache.get(wp_sym, {})
                has_structure = (
                    cached_entry.get("imports") or
                    cached_entry.get("exports") or
                    cached_entry.get("exchange")
                )

                # If cache is empty/incomplete, fetch full market details
                if not has_structure:
                    m_data = client.get_market(system, wp_sym)
                    if isinstance(m_data, dict) and "error" not in m_data:
                        _save_market_to_cache(wp_sym, m_data)
                        api_calls += 1
                        time.sleep(0.2)  # Rate limit kindness

        if api_calls > 0:
            console.print(f"  [dim green]Hydrated {api_calls} markets in {system}[/dim green]")


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
    console.rule(style="yellow")
    console.print()


# ──────────────────────────────────────────────
#  LLM decision cycle (one call + tool execution)
# ──────────────────────────────────────────────

def _run_llm_cycle(
    messages: list,
    context: NarrativeContext,
    fleet: FleetTracker,
    behavior_engine,
    llm_with_tools,
    active_tools: list,
    alert_text: str,
    iteration: int,
    debug: bool,
    action_queue: ActionQueue = None,
) -> Optional[list]:
    """
    Run one LLM decision-execute-narrate cycle.

    Returns the updated messages list, or None if the mission is complete.
    Mutates fleet and context as side effects.
    """
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

    # Build fresh game state, appending behavior status and any alerts
    game_state = gather_game_state(fleet, context)
    game_state += f"\n\n[Behavior Status]\n{behavior_engine.summary()}"
    if alert_text:
        game_state += f"\n\n[ALERTS]\n{alert_text}"
    messages.append(SystemMessage(content=game_state))

    display_thinking(messages)

    # Debug: dump full prompt
    if debug:
        console.print("\n[bold yellow]===== DEBUG: Full LLM Context =====[/bold yellow]")

        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = msg.content if hasattr(msg, 'content') else str(msg)

            if isinstance(msg, SystemMessage):
                if "=== CURRENT GAME STATE ===" in content:
                    console.print(f"[yellow]--- [{i}] CURRENT GAME STATE (injected) ---[/yellow]")
                    console.print(f"[cyan]{content}[/cyan]")
                else:
                    console.print(f"[yellow]--- [{i}] SYSTEM PROMPT ---[/yellow]")
                    console.print(f"[dim]{content}[/dim]")

            elif isinstance(msg, HumanMessage):
                console.print(f"[yellow]--- [{i}] INITIAL PROMPT ---[/yellow]")
                console.print(f"[dim]{content}[/dim]")

            elif isinstance(msg, AIMessage):
                console.print(f"[yellow]--- [{i}] AI DECISION ---[/yellow]")
                if content and content.strip():
                    console.print(f"[dim italic]{content}[/dim italic]")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', '?')
                        tool_args = tc.get('args', {})
                        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
                        console.print(f"[green]  → {tool_name}({args_str})[/green]")

            elif isinstance(msg, ToolMessage):
                tool_id = msg.tool_call_id if hasattr(msg, 'tool_call_id') else '?'
                tool_name = "?"
                if i > 0:
                    for j in range(i-1, -1, -1):
                        if isinstance(messages[j], AIMessage) and hasattr(messages[j], 'tool_calls'):
                            for tc in messages[j].tool_calls:
                                if tc.get('id') == tool_id:
                                    tool_name = tc.get('name', '?')
                                    break
                            if tool_name != "?":
                                break

                informational_tools = {"view_ships", "view_agent", "view_contracts", "list_ships"}
                if content == "OK":
                    console.print(f"[dim]  ← {tool_name}: OK[/dim]")
                elif "Error:" in content:
                    console.print(f"[red]  ← {tool_name} ERROR: {content}[/red]")
                elif tool_name in informational_tools:
                    console.print(f"[dim]  ← {tool_name}: [data in game state][/dim]")
                else:
                    console.print(f"[dim]  ← {tool_name}: {content}[/dim]")

        console.print(f"[yellow]--- Available tools: {', '.join(t.name for t in active_tools)} ---[/yellow]")
        console.print("[bold yellow]===== END DEBUG =====[/bold yellow]\n")

    # DECIDE — call LLM with tools
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        console.print(f"  [red]LLM error: {e}[/red]")
        return messages

    messages.append(response)

    if response.content or (hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('thinking')):
        display_decision(response)
        if response.content and "MISSION COMPLETE" in response.content.upper():
            console.print("\n  [bold green]✓ Mission complete![/bold green]")
            return None

    if not response.tool_calls:
        if not response.content:
            console.print("  [yellow]No action taken. Prompting to continue...[/yellow]")
            messages.append(HumanMessage(content="Please take an action or say MISSION COMPLETE if done."))
        return messages

    # EXECUTE — run each tool
    tool_results: list[tuple[str, str, bool]] = []
    has_significant_action = False

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        display_tool_call(tool_name, tool_args)

        ship_symbol = tool_args.get("ship_symbol") or tool_args.get("from_ship", "")
        ship_blocked = False
        if ship_symbol and tool_name in (WAITING_TOOLS | STATE_CHANGING_TOOLS):
            ship_status = fleet.get_ship(ship_symbol)
            if ship_status and not ship_status.is_available():
                # Ship is busy — check if it has a behavior
                has_behavior = ship_symbol in behavior_engine.behaviors
                if has_behavior:
                    secs = ship_status.seconds_until_available()
                    result = (f"Error: {ship_symbol} is busy ({ship_status.busy_reason}, {secs:.0f}s remaining) "
                              f"and has an active behavior. cancel_behavior first to operate manually.")
                    is_error = True
                    ship_blocked = True
                elif action_queue and tool_name in STATE_CHANGING_TOOLS:
                    # Queue the action instead of erroring
                    result = action_queue.enqueue(ship_symbol, tool_name, tool_args)
                    is_error = False
                    ship_blocked = True
                else:
                    secs = ship_status.seconds_until_available()
                    result = f"Error: {ship_symbol} is busy ({ship_status.busy_reason}, {secs:.0f}s remaining). Try another ship."
                    is_error = True
                    ship_blocked = True

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

        if is_error:
            write_event({"type": "tool_error", "tool": tool_name, "error": result})
        else:
            write_event({"type": "tool_result", "tool": tool_name, "result": result})

        tool_results.append((tool_name, result, is_error))

        if tool_name in SIGNIFICANT_TOOLS and not is_error:
            has_significant_action = True

        if tool_name in WAITING_TOOLS and not is_error:
            wait_time = get_last_wait(tool_name)
            if wait_time > 0:
                if ship_symbol:
                    if tool_name == "navigate_ship":
                        fleet.set_transit(ship_symbol, wait_time)
                    elif tool_name == "extract_ore":
                        fleet.set_extraction_cooldown(ship_symbol, wait_time)

        msg_content = result
        if not is_error and tool_name in REDUNDANT_RESULT_TOOLS:
            msg_content = "OK"
        messages.append(ToolMessage(
            content=msg_content,
            tool_call_id=tool_call["id"],
        ))

    # AUTO-DISCOVER: Check markets at ship locations after tool execution
    if tool_results:
        discovered = auto_discover_markets()
        tool_results.extend(discovered)

    # Refresh game state immediately after tool execution
    if tool_results:
        messages = [
            msg for msg in messages
            if not (isinstance(msg, SystemMessage) and
                    "=== CURRENT GAME STATE ===" in msg.content)
        ]
        console.print("  [dim]Refreshing game state after actions...[/dim]")
        fresh_state = gather_game_state(fleet, context)
        fresh_state += f"\n\n[Behavior Status]\n{behavior_engine.summary()}"
        if fresh_state:
            messages.append(SystemMessage(content=fresh_state))

    # NARRATE (optional)
    narrative_tool_results = [(name, result) for name, result, is_err in tool_results if not is_err]
    fleet_state = fleet.fleet_summary() if has_significant_action else ""

    if ENABLE_PER_ACTION_NARRATIVE and has_significant_action and narrative_tool_results:
        console.print("  [dim]📝 Generating log entry...[/dim]")
        segment = generate_narrative(narrative_tool_results, context, MODEL, fleet_state)
        if segment:
            context.add_segment(segment)
            context.persist()
            context.persist_full()
            display_narrative(segment, context)

    return messages


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
    behavior_engine = get_behavior_engine()

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

            # Remove any old system prompts from loaded session
            messages = [
                msg for msg in messages
                if not (isinstance(msg, SystemMessage) and msg.content.strip().startswith("You are WHATER"))
            ]

            # Always prepend the latest SYSTEM_PROMPT
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

            # Fetch and inject FRESH game state on resume
            fresh_state = gather_game_state(fleet, context)
            if fresh_state:
                console.print(f"  [dim]Injected fresh game state[/dim]")
                messages.append(SystemMessage(content=fresh_state))

            # Discover all marketplace waypoints in all known systems
            console.print("  [dim]Discovering all marketplaces in known systems...[/dim]")
            discover_all_markets(fleet)
            auto_discover_markets()

            # Re-gather game state to include newly discovered markets
            fresh_state = gather_game_state(fleet, context)
            if fresh_state:
                # Replace the previous game state with updated one
                messages = [msg for msg in messages if not (isinstance(msg, SystemMessage) and "=== CURRENT GAME STATE ===" in msg.content)]
                messages.append(SystemMessage(content=fresh_state))
            console.print()

    # Build initial messages if not resuming
    if messages is None:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]

        # Gather and inject full game state upfront
        console.print("  [dim]Gathering game state...[/dim]")
        fresh_state = gather_game_state(fleet, context)
        if fresh_state:
            messages.append(SystemMessage(content=fresh_state))

        # Discover all marketplace waypoints in all known systems
        console.print("  [dim]Discovering all marketplaces in known systems...[/dim]")
        discover_all_markets(fleet)
        auto_discover_markets()

        # Re-gather game state to include newly discovered markets
        fresh_state = gather_game_state(fleet, context)
        if fresh_state:
            # Replace the previous game state with updated one
            messages = [msg for msg in messages if not (isinstance(msg, SystemMessage) and "=== CURRENT GAME STATE ===" in msg.content)]
            messages.append(SystemMessage(content=fresh_state))

        messages.append(HumanMessage(
            content="Review [Behavior Status]. Assign behaviors to any IDLE ships. If all ships are already assigned, check [ALERTS] and handle any that are present."
        ))

    # Always run strategic reflection on startup to ensure fresh plan
    if not load_plan():  # Only if no plan exists
        console.print("\n  [yellow]🔮 No plan found - running strategic reflection...[/yellow]")
        game_state = gather_game_state(fleet, context)
        reflection_segment, reflection_data = generate_strategic_reflection(context, game_state, MODEL)

        if reflection_segment:
            context.add_segment(reflection_segment)
            context.persist()
            context.persist_full()

            # Write recommended plan to plan.txt
            if reflection_data and "recommended_plan" in reflection_data:
                recommended_plan = reflection_data["recommended_plan"]
                if recommended_plan:
                    from tools import update_plan
                    result = update_plan.invoke({"plan": recommended_plan})
                    console.print(f"  [dim green]📋 Initial plan created: {result}[/dim green]")

    iteration = start_iteration
    alert_queue: list[str] = []
    action_queue = ActionQueue()
    last_strategy_time = time.time()

    while True:
        # ── STATE SYNC ────────────────────────────────────────────────────
        # Check if play_cli.py modified behaviors.json and reload if needed
        behavior_engine.sync_state()
        # ── QUEUED ACTIONS ────────────────────────────────────────────────
        # Execute queued actions for ships that just became available.
        for ship_symbol in list(fleet.ships):
            ship_status = fleet.get_ship(ship_symbol)
            if ship_status and ship_status.is_available() and action_queue.has_queued(ship_symbol):
                action = action_queue.get_ready(ship_symbol)
                if action:
                    tool_name, tool_args = action
                    console.print(f"  [dim green]Executing queued {tool_name} for {ship_symbol}[/dim green]")
                    tool_func = get_tool_by_name(tool_name)
                    if tool_func:
                        try:
                            result = tool_func.invoke(tool_args)
                            display_tool_result(tool_name, result, "Error:" in result)
                            # Track wait times for navigation/extraction
                            if tool_name in WAITING_TOOLS and "Error:" not in result:
                                wait_time = get_last_wait(tool_name)
                                if wait_time > 0 and ship_symbol:
                                    if tool_name == "navigate_ship":
                                        fleet.set_transit(ship_symbol, wait_time)
                                    elif tool_name == "extract_ore":
                                        fleet.set_extraction_cooldown(ship_symbol, wait_time)
                        except Exception as e:
                            console.print(f"    [red]Queued {tool_name} failed: {e}[/red]")

        # ── TACTICAL LAYER ────────────────────────────────────────────────
        # Tick all ship behaviors. Fast — no LLM involved.
        for ship_symbol in list(fleet.ships):
            alert = behavior_engine.tick(ship_symbol, fleet, client)
            # Log behavior step execution
            cfg = behavior_engine.behaviors.get(ship_symbol)
            if cfg and cfg.last_action:
                status_color = "red" if "ERROR" in cfg.last_action else "dim green"
                console.print(f"  [{status_color}]⚙ {ship_symbol}: {cfg.last_action}[/{status_color}]")
            if alert and alert not in alert_queue:
                alert_queue.append(alert)

        # ── STRATEGIC LAYER ───────────────────────────────────────────────
        # Call the LLM only when something needs attention.
        idle_ships = behavior_engine.get_idle_ships(fleet)

        # Check if it's time for a periodic strategic review
        time_since_strategy = time.time() - last_strategy_time
        strategic_review_needed = time_since_strategy > STRATEGY_INTERVAL

        has_alerts = bool(alert_queue)

        # Wake up if: Alerts OR Idle Ships OR Strategic Timer
        if ENABLE_LLM and (has_alerts or idle_ships or strategic_review_needed):
            # If waking up strictly for strategy, inject a prompt
            if strategic_review_needed and not has_alerts and not idle_ships:
                console.print(f"\n  [dim magenta]⏰ Periodic Strategic Review ({int(time_since_strategy)}s elapsed)[/dim magenta]")
                # We add this to alert_queue so it gets passed to the LLM context
                alert_queue.append(
                    f"PERIODIC_STRATEGIC_REVIEW: It has been {int(time_since_strategy/60)} minutes since the last review. "
                    "Analyze fleet efficiency. Are there better trade routes? Should you update the plan?"
                )
                # Reset timer
                last_strategy_time = time.time()
                # Update flag since we modified alert_queue
                has_alerts = True

            if idle_ships:
                console.print(f"  [dim]Idle ships (no behavior): {', '.join(idle_ships)}[/dim]")

            alert_text = "\n".join(alert_queue)
            alert_queue.clear()

            console.print(f"\n[dim]─── Cycle {iteration + 1} ───[/dim]")

            result = _run_llm_cycle(
                messages=messages,
                context=context,
                fleet=fleet,
                behavior_engine=behavior_engine,
                llm_with_tools=llm_with_tools,
                active_tools=active_tools,
                alert_text=alert_text,
                iteration=iteration,
                debug=debug,
                action_queue=action_queue,
            )

            if result is None:
                break  # MISSION COMPLETE

            messages = result
            iteration += 1
            save_session(messages, iteration)

            # Reset heartbeat timer if the LLM actually ran
            last_strategy_time = time.time()

        time.sleep(TICK_INTERVAL)

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
        Path("behaviors.json").unlink(missing_ok=True)
        Path("plan.txt").unlink(missing_ok=True)
        Path("market_cache.json").unlink(missing_ok=True)
        print("Cleared all saved state.")
        sys.exit(0)

    print(f"SpaceTraders bot starting (model: {MODEL})")
    run_agent(fresh_start=args.fresh, debug=args.debug)
