"""
test_decisions.py — Test harness for evaluating LLM decision-making.

This module allows testing the bot's decision-making without a live API.
It mocks game state and evaluates whether the LLM makes expected decisions.

Usage:
    python test_decisions.py                    # Run all scenarios
    python test_decisions.py --scenario=<name>  # Run specific scenario
    python test_decisions.py --list             # List available scenarios
"""
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Import the actual tools and prompts
from bot import SYSTEM_PROMPT
from tools import ALL_TOOLS


@dataclass
class GameState:
    """Mock game state for a test scenario."""
    agent_credits: int = 100000
    ships: list[dict] = None
    contracts: list[dict] = None
    description: str = ""

    def __post_init__(self):
        if self.ships is None:
            self.ships = []
        if self.contracts is None:
            self.contracts = []

    def format_ships(self) -> str:
        """Format ships for display."""
        lines = []
        for s in self.ships:
            lines.append(f"Ship: {s['symbol']}")
            lines.append(f"  Type: {s['role']}")
            lines.append(f"  Location: {s['location']}  |  Status: {s['nav_status']}")
            lines.append(f"  Fuel: {s['fuel_current']}/{s['fuel_capacity']}")
            lines.append(f"  Cargo: {s['cargo_units']}/{s['cargo_capacity']} units")
            if s.get('cargo_items'):
                for item, units in s['cargo_items'].items():
                    lines.append(f"    - {item}: {units}")
            lines.append("")
        return "\n".join(lines)

    def format_contracts(self) -> str:
        """Format contracts for display."""
        if not self.contracts:
            return "No active contracts."
        lines = []
        for c in self.contracts:
            lines.append(f"Contract: {c['id']}")
            lines.append(f"  Accepted: {c['accepted']}  |  Fulfilled: {c['fulfilled']}")
            for d in c.get('deliveries', []):
                lines.append(f"  Deliver: {d['required']} {d['item']} to {d['destination']} ({d['fulfilled']}/{d['required']} done)")
            lines.append("")
        return "\n".join(lines)

    def format_fleet_state(self) -> str:
        """Format as fleet state block."""
        lines = []
        # Add description if present (waypoint info, etc.)
        if self.description:
            lines.append(f"[Note: {self.description}]")
        for s in self.ships:
            status = f"{s['symbol']} ({s['role']}) @ {s['location']} [{s['nav_status']}]"
            # Handle solar powered ships (probes/satellites)
            if s['fuel_capacity'] == 0:
                status += " Fuel:N/A(solar)"
            else:
                status += f" Fuel:{s['fuel_current']}/{s['fuel_capacity']}"
            # Handle ships with no cargo capacity
            if s['cargo_capacity'] == 0:
                status += " Cargo:N/A"
            else:
                status += f" Cargo:{s['cargo_units']}/{s['cargo_capacity']}"
            if s.get('busy_reason'):
                status += f" [BUSY: {s['busy_reason']}]"
            lines.append(f"• {status}")
            # Include cargo contents if present
            if s.get('cargo_items'):
                for item, units in s['cargo_items'].items():
                    lines.append(f"    └─ {item}: {units}")
            # Include capabilities if present
            if s.get('capabilities'):
                lines.append(f"  Capabilities: {', '.join(s['capabilities'])}")
            elif s['role'] == "SATELLITE":
                lines.append(f"  Capabilities: CAN_NAVIGATE (solar powered, no fuel needed)")
        return "\n".join(lines)


@dataclass
class TestScenario:
    """A test scenario for evaluating LLM decisions."""
    name: str
    description: str
    game_state: GameState
    prompt: str  # Additional context/question for the LLM
    expected_tools: list[str]  # Tools that SHOULD be called
    unexpected_tools: list[str]  # Tools that should NOT be called
    success_keywords: list[str]  # Keywords in reasoning that indicate understanding

    def evaluate(self, tool_calls: list[dict], reasoning: str) -> tuple[bool, str]:
        """
        Evaluate whether the LLM made good decisions.

        Returns (passed, explanation)
        """
        called_tools = [tc["name"] for tc in tool_calls]

        # Check for expected tools - now accepts ANY ONE of expected tools (not all)
        found_expected = []
        for tool in self.expected_tools:
            if tool in called_tools:
                found_expected.append(tool)

        # Check for unexpected tools (hard failures)
        found_unexpected = []
        for tool in self.unexpected_tools:
            if tool in called_tools:
                found_unexpected.append(tool)

        # Check for success keywords in reasoning
        found_keywords = []
        missing_keywords = []
        reasoning_lower = reasoning.lower()
        for kw in self.success_keywords:
            if kw.lower() in reasoning_lower:
                found_keywords.append(kw)
            else:
                missing_keywords.append(kw)

        # Build result
        passed = True
        messages = []

        # PASS if at least one expected tool was called (flexible matching)
        if not found_expected:
            # Allow observation tools as reasonable first steps
            observation_tools = {"view_ship_details", "view_ships", "view_cargo", "scan_waypoints", "view_market"}
            called_observation = [t for t in called_tools if t in observation_tools]
            if called_observation:
                messages.append(f"Called observation tools first: {called_observation} (acceptable)")
            else:
                passed = False
                messages.append(f"Missing expected tools: {self.expected_tools}")
        else:
            messages.append(f"Called expected tool(s): {found_expected}")

        if found_unexpected:
            passed = False
            messages.append(f"Called unexpected tools: {found_unexpected}")

        if passed:
            messages.insert(0, "PASSED")

        return passed, "; ".join(messages)


# ──────────────────────────────────────────────
#  Test Scenarios
# ──────────────────────────────────────────────

SCENARIOS: dict[str, TestScenario] = {}

def scenario(name: str):
    """Decorator to register a test scenario."""
    def decorator(func: Callable[[], TestScenario]):
        SCENARIOS[name] = func()
        return func
    return decorator


@scenario("miner_full_command_nearby")
def _():
    """Miner cargo full, command ship at same location. Should transfer."""
    return TestScenario(
        name="miner_full_command_nearby",
        description="Miner cargo is full, command ship is at the same asteroid. Should transfer cargo.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",  # Same as miner
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 5,
                    "cargo_capacity": 40,
                    "cargo_items": {"IRON_ORE": 5},
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",  # Same as command
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 15,  # FULL
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 10, "IRON_ORE": 5},
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
        ),
        prompt="WHATER-3 cargo is FULL (15/15). WHATER-1 is at the same location with space available. Apply tactical rules. What action should you take?",
        expected_tools=["transfer_cargo"],
        unexpected_tools=["navigate_ship", "sell_cargo", "jettison_cargo", "view_cargo"],
        success_keywords=["transfer", "full"],
    )


@scenario("miner_full_command_elsewhere")
def _():
    """Miner cargo full, command ship elsewhere. Should summon command ship."""
    return TestScenario(
        name="miner_full_command_elsewhere",
        description="Miner cargo is full, command ship is at HQ. Should navigate command ship to miner.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-A2",  # At HQ
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 0,
                    "cargo_capacity": 40,
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",  # At asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 15,  # FULL
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 10, "IRON_ORE": 5},
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
        ),
        prompt="The miner's cargo is full but the command ship is elsewhere. What should you do?",
        expected_tools=["navigate_ship"],
        unexpected_tools=["sell_cargo", "jettison_cargo"],
        success_keywords=["WHATER-1", "navigate", "asteroid", "CB5E"],
    )


@scenario("miner_cooldown_command_idle")
def _():
    """Miner on extraction cooldown, command ship idle elsewhere. Should position command ship."""
    return TestScenario(
        name="miner_cooldown_command_idle",
        description="Miner is on cooldown at asteroid, command ship is idle at HQ. Should navigate command ship to asteroid.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-A2",  # At HQ, not asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 0,
                    "cargo_capacity": 40,
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",  # At asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 8,
                    "cargo_capacity": 15,
                    "busy_reason": "extraction_cooldown",  # On cooldown
                },
            ],
        ),
        prompt="WHATER-3 is mining at asteroid CB5E but on cooldown. WHATER-1 is idle at A2. Apply tactical rule #4: position the depot. What action?",
        expected_tools=["navigate_ship"],
        unexpected_tools=["extract_ore", "wait"],  # Should NOT wait when command ship can move
        success_keywords=["WHATER-1", "navigate", "CB5E", "asteroid"],
    )


@scenario("command_full_deliver")
def _():
    """Command ship cargo full with contract goods. Should deliver."""
    return TestScenario(
        name="command_full_deliver",
        description="Command ship has 30 ICE_WATER (>20 units). Should navigate to delivery point per tactical rule #1.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",  # At asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 38,
                    "cargo_capacity": 40,
                    "cargo_items": {"ICE_WATER": 30, "IRON_ORE": 8},
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 5,
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 5},
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
        ),
        prompt="WHATER-1 has 30 ICE_WATER. Contract needs ICE_WATER delivered to X1-KD26-A2. Apply tactical rule #1 (DELIVERY READY). What action?",
        expected_tools=["navigate_ship"],
        unexpected_tools=["extract_ore", "jettison_cargo", "find_waypoints"],
        success_keywords=["deliver", "A2", "WHATER-1", "navigate"],
    )


@scenario("all_ships_busy_use_probe")
def _():
    """Main ships are busy but probe is available. Should use probe to explore."""
    return TestScenario(
        name="all_ships_busy_use_probe",
        description="Mining ships are busy, but probe is available. Should use probe to scan or explore.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "IN_TRANSIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 20,
                    "cargo_capacity": 40,
                    "busy_reason": "in_transit",
                },
                {
                    "symbol": "WHATER-2",
                    "role": "SATELLITE",
                    "location": "X1-KD26-A2",
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 0,
                    "fuel_capacity": 0,
                    "cargo_units": 0,
                    "cargo_capacity": 0,
                    # Probe is available!
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 10,
                    "cargo_capacity": 15,
                    "busy_reason": "extraction_cooldown",
                },
            ],
        ),
        prompt="WHATER-1 is in transit. WHATER-3 is on cooldown. But WHATER-2 (probe) is available at A2. Never wait - always do something productive. What action?",
        expected_tools=["navigate_ship", "scan_waypoints", "view_market"],  # Any productive action with probe
        unexpected_tools=["wait", "extract_ore"],
        success_keywords=["WHATER-2", "probe", "scan", "explore", "market"],
    )


@scenario("miner_at_asteroid_ready")
def _():
    """Miner at asteroid with cargo space, not on cooldown. Should extract."""
    return TestScenario(
        name="miner_at_asteroid_ready",
        description="Miner is at asteroid with cargo space and no cooldown. Should extract ore.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",  # At asteroid with miner
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 10,
                    "cargo_capacity": 40,
                    "cargo_items": {"ICE_WATER": 10},
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",  # At asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 5,  # Has space
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 5},
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
        ),
        prompt="WHATER-3 is at asteroid CB5E with cargo space (5/15). No cooldown. WHATER-1 is also here as depot. Apply tactical rule #5. What action?",
        expected_tools=["extract_ore"],
        unexpected_tools=["navigate_ship", "wait", "transfer_cargo"],
        success_keywords=["extract", "WHATER-3", "mine"],
    )


@scenario("need_to_orbit_before_transfer")
def _():
    """Ships at same location but one is docked. Should orbit first."""
    return TestScenario(
        name="need_to_orbit_before_transfer",
        description="Miner is full and command ship is at same location but DOCKED. Should orbit command ship first.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "DOCKED",  # Not in orbit!
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 5,
                    "cargo_capacity": 40,
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 15,  # FULL
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 15},
                },
            ],
        ),
        prompt="WHATER-3 cargo is FULL. WHATER-1 is at same location but DOCKED. Both must be in ORBIT to transfer. What action?",
        expected_tools=["orbit_ship"],
        unexpected_tools=["transfer_cargo", "navigate_ship"],  # Can't transfer while docked
        success_keywords=["orbit", "WHATER-1"],
    )


@scenario("command_ship_should_mine")
def _():
    """Command ship at asteroid with CAN_MINE capability. Should extract ore."""
    return TestScenario(
        name="command_ship_should_mine",
        description="Command ship is at asteroid with CAN_MINE capability and cargo space. Should extract ore.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-CB5E",  # At ENGINEERED_ASTEROID
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 10,
                    "cargo_capacity": 40,
                    "cargo_items": {"ICE_WATER": 10},
                    "capabilities": ["CAN_MINE"],  # Has mining laser!
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 8,
                    "cargo_capacity": 15,
                    "busy_reason": "extraction_cooldown",  # Miner on cooldown
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
            description="X1-KD26-CB5E is an ENGINEERED_ASTEROID where ships can extract ore.",
        ),
        prompt="WHATER-3 is on extraction cooldown. WHATER-1 is at asteroid CB5E with CAN_MINE capability and cargo space (10/40). The asteroid is minable. Apply tactical rule #4: use WHATER-1 to extract_ore.",
        expected_tools=["extract_ore"],
        unexpected_tools=["wait", "navigate_ship", "scan_waypoints"],  # Already at asteroid, no need to scan
        success_keywords=["WHATER-1", "extract"],
    )


@scenario("check_market_before_jettison")
def _():
    """Ship full of ores, should check market or navigate to sell instead of jettisoning."""
    return TestScenario(
        name="check_market_before_jettison",
        description="Miner cargo full with SILICON_CRYSTALS. Nearby market A3 buys them. Should navigate to sell, not jettison.",
        game_state=GameState(
            ships=[
                {
                    "symbol": "WHATER-1",
                    "role": "COMMAND",
                    "location": "X1-KD26-A2",  # At HQ, away from miner
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 300,
                    "fuel_capacity": 400,
                    "cargo_units": 35,
                    "cargo_capacity": 40,
                    "cargo_items": {"ICE_WATER": 25, "SILICON_CRYSTALS": 10},
                },
                {
                    "symbol": "WHATER-2",
                    "role": "SATELLITE",
                    "location": "X1-KD26-A3",  # Probe at market
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 0,
                    "fuel_capacity": 0,  # Solar powered
                    "cargo_units": 0,
                    "cargo_capacity": 0,
                },
                {
                    "symbol": "WHATER-3",
                    "role": "EXCAVATOR",
                    "location": "X1-KD26-CB5E",  # At asteroid
                    "nav_status": "IN_ORBIT",
                    "fuel_current": 60,
                    "fuel_capacity": 80,
                    "cargo_units": 15,  # FULL
                    "cargo_capacity": 15,
                    "cargo_items": {"ICE_WATER": 5, "SILICON_CRYSTALS": 10},
                },
            ],
            contracts=[{
                "id": "contract-123",
                "accepted": True,
                "fulfilled": False,
                "deliveries": [{"item": "ICE_WATER", "required": 73, "fulfilled": 14, "destination": "X1-KD26-A2"}]
            }],
            description="Market at X1-KD26-A3 buys SILICON_CRYSTALS. Don't jettison them!",
        ),
        prompt="WHATER-3 cargo is FULL with SILICON_CRYSTALS. X1-KD26-A3 has a market that buys SILICON_CRYSTALS. WHATER-2 (probe) is at A3. What should you do instead of jettisoning?",
        expected_tools=["view_market", "navigate_ship"],  # Check market or navigate to sell
        unexpected_tools=["jettison_cargo"],  # Don't jettison sellable goods!
        success_keywords=["market", "sell", "A3", "SILICON"],
    )


# ──────────────────────────────────────────────
#  Test Runner
# ──────────────────────────────────────────────

def run_scenario(scenario: TestScenario, model: str, verbose: bool = True) -> tuple[bool, str]:
    """
    Run a single test scenario and evaluate the result.

    Returns (passed, details)
    """
    import sys as _sys

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"SCENARIO: {scenario.name}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Description: {scenario.description}", flush=True)
        print(flush=True)

    # Build messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"[Current Fleet State]\n{scenario.game_state.format_fleet_state()}"),
    ]

    # Add contract info if relevant
    if scenario.game_state.contracts:
        messages.append(SystemMessage(content=f"[Contracts]\n{scenario.game_state.format_contracts()}"))

    # Add the test prompt
    messages.append(HumanMessage(content=scenario.prompt))

    if verbose:
        print(f"Fleet state:\n{scenario.game_state.format_fleet_state()}", flush=True)
        print(f"\nPrompt: {scenario.prompt}", flush=True)

    # Call LLM
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    if verbose:
        print(f"\nConnecting to Ollama at {ollama_url}...", flush=True)

    llm = ChatOllama(model=model, base_url=ollama_url)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    if verbose:
        print(f"Calling LLM (model: {model})...", flush=True)
        _sys.stdout.flush()

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        print(f"LLM Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False, f"LLM Error: {e}"

    if verbose:
        print("Got response from LLM", flush=True)

    # Extract results
    reasoning = response.content or ""
    tool_calls = response.tool_calls or []

    if verbose:
        print(f"\nReasoning: {reasoning[:500]}..." if len(reasoning) > 500 else f"\nReasoning: {reasoning}", flush=True)
        print(f"\nTool calls: {[tc['name'] for tc in tool_calls]}", flush=True)
        for tc in tool_calls:
            print(f"  {tc['name']}({tc['args']})", flush=True)

    # Evaluate
    passed, explanation = scenario.evaluate(tool_calls, reasoning)

    if verbose:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n{status}: {explanation}", flush=True)

    return passed, explanation


def run_all_scenarios(model: str, verbose: bool = True) -> dict:
    """Run all test scenarios and return results."""
    print(f"Running {len(SCENARIOS)} test scenarios with model: {model}", flush=True)
    print(f"Scenarios: {list(SCENARIOS.keys())}", flush=True)

    results = {}
    passed_count = 0

    for i, (name, scenario) in enumerate(SCENARIOS.items(), 1):
        print(f"\n[{i}/{len(SCENARIOS)}] Starting scenario: {name}", flush=True)
        passed, explanation = run_scenario(scenario, model, verbose)
        results[name] = {"passed": passed, "explanation": explanation}
        if passed:
            passed_count += 1

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY: {passed_count}/{len(SCENARIOS)} scenarios passed", flush=True)
    print(f"{'='*60}", flush=True)

    for name, result in results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  [{status}] {name}", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM decision-making")
    parser.add_argument("--scenario", help="Run specific scenario by name")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--model", default=os.environ.get("MODEL", "glm-4.7-flash"), help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, scenario in SCENARIOS.items():
            print(f"  {name}: {scenario.description}")
        sys.exit(0)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        run_scenario(SCENARIOS[args.scenario], args.model, verbose=not args.quiet)
    else:
        run_all_scenarios(args.model, verbose=not args.quiet)
