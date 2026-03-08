"""
tool_generator.py — Auto-generate LangChain tools from the SpaceTraders API.

This module provides utilities to:
1. Generate tools from snisp library methods
2. Parse the OpenAPI spec to create tools directly
3. Dynamically expose new API endpoints as they're discovered

Usage:
    from tool_generator import generate_ship_tools, generate_all_tools

    tools = generate_all_tools(agent)
"""

import inspect
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.tools import tool

# ──────────────────────────────────────────────
#  OpenAPI Spec Parser
# ──────────────────────────────────────────────


def load_openapi_spec(path: str = "st.json") -> dict:
    """Load the SpaceTraders OpenAPI spec."""
    return json.loads(Path(path).read_text())


def extract_endpoints(spec: dict) -> list[dict]:
    """
    Extract all endpoints from the OpenAPI spec.

    Returns list of dicts with: path, method, operationId, summary, description, parameters
    """
    endpoints = []

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method in ("get", "post", "put", "patch", "delete"):
                endpoint = {
                    "path": path,
                    "method": method.upper(),
                    "operationId": details.get("operationId", ""),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "parameters": details.get("parameters", []),
                    "requestBody": details.get("requestBody", {}),
                    "tags": details.get("tags", []),
                }
                endpoints.append(endpoint)

    return endpoints


def categorize_endpoints(endpoints: list[dict]) -> dict[str, list[dict]]:
    """Group endpoints by their primary tag."""
    categories = {}
    for ep in endpoints:
        tag = ep["tags"][0] if ep["tags"] else "misc"
        if tag not in categories:
            categories[tag] = []
        categories[tag].append(ep)
    return categories


def print_api_coverage(spec_path: str = "st.json"):
    """Print a summary of all API endpoints for review."""
    spec = load_openapi_spec(spec_path)
    endpoints = extract_endpoints(spec)
    categories = categorize_endpoints(endpoints)

    print(f"Total endpoints: {len(endpoints)}\n")

    for tag, eps in sorted(categories.items()):
        print(f"\n=== {tag} ({len(eps)} endpoints) ===")
        for ep in eps:
            print(f"  {ep['method']:6} {ep['path']}")
            print(f"         {ep['summary']}")


# ──────────────────────────────────────────────
#  Tool Generation from snisp
# ──────────────────────────────────────────────


def make_ship_tool(
    method_name: str, method: Callable, ship_getter: Callable
) -> Callable:
    """
    Create a LangChain tool from a Ship method.

    Args:
        method_name: Name of the method (e.g., "extract", "navigate")
        method: The bound method from Ship class
        ship_getter: Function that takes ship_symbol and returns Ship object
    """
    sig = inspect.signature(method)
    params = list(sig.parameters.keys())

    # Build docstring from method
    doc = method.__doc__ or f"Call {method_name} on a ship."

    @tool
    def ship_tool(ship_symbol: str, **kwargs) -> str:
        """Dynamic ship tool."""
        try:
            ship = ship_getter(ship_symbol)
            result = getattr(ship, method_name)(**kwargs)
            return format_result(result)
        except Exception as e:
            return f"Error: {e}"

    # Update the tool's name and docstring
    ship_tool.__name__ = method_name
    ship_tool.__doc__ = (
        f"{doc}\n\nArgs:\n  ship_symbol: The ship to use (e.g., 'WHATER-1')"
    )

    return ship_tool


def format_result(result: Any) -> str:
    """Format a snisp result object for display."""
    if result is None:
        return "Success (no data returned)"
    if isinstance(result, str):
        return result
    if hasattr(result, "to_dict"):
        return json.dumps(result.to_dict(), indent=2)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return "Empty list"
        if hasattr(result[0], "to_dict"):
            return json.dumps([r.to_dict() for r in result[:10]], indent=2)
        return str(result[:10])
    return str(result)


# ──────────────────────────────────────────────
#  Ship Method Wrappers
# ──────────────────────────────────────────────

# These are hand-crafted tool definitions that wrap snisp Ship methods
# with proper LangChain tool signatures.

SHIP_TOOL_CONFIGS = [
    # Navigation
    {
        "name": "ship_navigate",
        "method": "navigate",
        "description": "Navigate ship to a waypoint. Ship must be in orbit.",
        "params": {"waypoint": "Waypoint symbol to navigate to (e.g., 'X1-KD26-CB5E')"},
    },
    {
        "name": "ship_orbit",
        "method": "orbit",
        "description": "Put ship into orbit at current waypoint. Required before navigation or extraction.",
        "params": {},
    },
    {
        "name": "ship_dock",
        "method": "dock",
        "description": "Dock ship at current waypoint. Required before refueling or selling.",
        "params": {},
    },
    {
        "name": "ship_jump",
        "method": "jump",
        "description": "Jump ship to another system using a jump gate.",
        "params": {"system": "Target system symbol"},
    },
    {
        "name": "ship_warp",
        "method": "warp",
        "description": "Warp ship to a waypoint in another system (uses antimatter).",
        "params": {"waypoint": "Target waypoint symbol"},
    },
    # Mining & Resources
    {
        "name": "ship_extract",
        "method": "extract",
        "description": "Extract resources from an asteroid. Ship must be in orbit at an asteroid.",
        "params": {},
    },
    {
        "name": "ship_survey",
        "method": "survey",  # Note: This is on waypoints, not ship
        "description": "Survey the current location to find resource deposits.",
        "params": {},
    },
    {
        "name": "ship_siphon",
        "method": "siphon",
        "description": "Siphon gas from a gas giant. Ship must have siphon capability.",
        "params": {},
    },
    {
        "name": "ship_refine",
        "method": "refine",
        "description": "Refine raw ore or gas into refined materials.",
        "params": {"produce": "What to produce (e.g., 'IRON', 'FUEL')"},
    },
    # Trading
    {
        "name": "ship_sell",
        "method": "sell",
        "description": "Sell cargo at the current market. Ship must be docked.",
        "params": {
            "item": "Trade symbol (e.g., 'IRON_ORE')",
            "units": "Number of units to sell",
        },
    },
    {
        "name": "ship_purchase",
        "method": "purchase",
        "description": "Purchase cargo from the current market. Ship must be docked.",
        "params": {
            "item": "Trade symbol (e.g., 'FUEL')",
            "units": "Number of units to buy",
        },
    },
    {
        "name": "ship_transfer",
        "method": "transfer",
        "description": "Transfer cargo to another ship at the same waypoint.",
        "params": {
            "item": "Trade symbol",
            "units": "Number of units",
            "to_ship": "Target ship symbol",
        },
    },
    {
        "name": "ship_jettison",
        "method": "jettison",
        "description": "Jettison cargo into space. Use to make room.",
        "params": {"item": "Trade symbol", "units": "Number of units to discard"},
    },
    # Ship Management
    {
        "name": "ship_refuel",
        "method": "refuel",
        "description": "Refuel ship at the current waypoint. Must be docked at a fuel station.",
        "params": {},
    },
    {
        "name": "ship_repair",
        "method": "repair",
        "description": "Repair ship at a shipyard. Ship must be docked.",
        "params": {},
    },
    {
        "name": "ship_scan",
        "method": "scan",
        "description": "Scan for ships, waypoints, or systems from current location.",
        "params": {"mode": "What to scan: 'ships', 'waypoints', or 'systems'"},
    },
    {
        "name": "ship_install_mount",
        "method": "install_mount",
        "description": "Install a mount (weapon, mining laser, etc.) on the ship.",
        "params": {"mount": "Mount symbol to install"},
    },
    {
        "name": "ship_remove_mount",
        "method": "remove_mount",
        "description": "Remove a mount from the ship.",
        "params": {"mount": "Mount symbol to remove"},
    },
]


def generate_snisp_tools(agent) -> list:
    """
    Generate LangChain tools from a snisp Agent.

    Args:
        agent: A snisp Agent instance (from snisp.agent.Agent)

    Returns:
        List of LangChain tool functions
    """
    tools = []

    # Build a ship lookup
    ships_by_symbol = {}

    def refresh_ships():
        """Refresh the ship cache."""
        nonlocal ships_by_symbol
        ships_by_symbol = {}
        for ship in agent.fleet.ships:
            ships_by_symbol[ship.symbol] = ship

    def get_ship(symbol: str):
        """Get a ship by symbol, refreshing if needed."""
        if symbol not in ships_by_symbol:
            refresh_ships()
        if symbol not in ships_by_symbol:
            raise ValueError(f"Ship {symbol} not found in fleet")
        return ships_by_symbol[symbol]

    # Generate ship operation tools
    for config in SHIP_TOOL_CONFIGS:
        tool_func = create_ship_tool(config, get_ship)
        if tool_func:
            tools.append(tool_func)

    # Add fleet-level tools
    @tool
    def list_ships() -> str:
        """List all ships in your fleet with their status, location, and cargo."""
        refresh_ships()
        lines = []
        for symbol, ship in sorted(ships_by_symbol.items()):
            lines.append(f"Ship: {symbol}")
            lines.append(f"  Role: {ship.registration.role}")
            lines.append(f"  Location: {ship.nav.waypoint_symbol} ({ship.nav.status})")
            lines.append(f"  Fuel: {ship.fuel.current}/{ship.fuel.capacity}")
            cargo = ship.cargo
            lines.append(f"  Cargo: {cargo.units}/{cargo.capacity}")
            if cargo.inventory:
                for item in cargo.inventory:
                    lines.append(f"    - {item.symbol}: {item.units}")
            lines.append("")
        return "\n".join(lines) if lines else "No ships in fleet"

    tools.append(list_ships)

    @tool
    def list_contracts() -> str:
        """List all contracts with their requirements and progress."""
        lines = []
        for contract in agent.contracts:
            lines.append(f"Contract: {contract.id}")
            lines.append(f"  Type: {contract.type}")
            lines.append(
                f"  Accepted: {contract.accepted}, Fulfilled: {contract.fulfilled}"
            )
            terms = contract.terms
            lines.append(
                f"  Payment: {terms.payment.on_accepted} on accept, {terms.payment.on_fulfilled} on fulfill"
            )
            for deliver in terms.deliver:
                lines.append(
                    f"  Deliver: {deliver.units_required} {deliver.trade_symbol} to {deliver.destination_symbol}"
                )
                lines.append(
                    f"           ({deliver.units_fulfilled}/{deliver.units_required} done)"
                )
            lines.append("")
        return "\n".join(lines) if lines else "No contracts"

    tools.append(list_contracts)

    @tool
    def agent_status() -> str:
        """Get current agent status: credits, headquarters, ship count."""
        return (
            f"Agent: {agent.symbol}\n"
            f"Credits: {agent.credits}\n"
            f"Headquarters: {agent.headquarters}\n"
            f"Ship count: {agent.ship_count}"
        )

    tools.append(agent_status)

    return tools


def create_ship_tool(config: dict, get_ship: Callable):
    """Create a single ship tool from config."""
    method_name = config["method"]
    tool_name = config["name"]
    description = config["description"]
    params = config.get("params", {})

    if not params:
        # Simple method, no extra params
        @tool
        def ship_method(ship_symbol: str) -> str:
            """Ship method wrapper."""
            try:
                ship = get_ship(ship_symbol)
                result = getattr(ship, method_name)()
                return format_result(result)
            except Exception as e:
                return f"Error: {e}"

        ship_method.__name__ = tool_name
        ship_method.__doc__ = (
            f"{description}\n\nArgs:\n  ship_symbol: Ship to use (e.g., 'WHATER-1')"
        )
        return ship_method

    elif len(params) == 1:
        param_name, param_desc = list(params.items())[0]

        @tool
        def ship_method_1arg(ship_symbol: str, arg1: str) -> str:
            """Ship method wrapper with 1 arg."""
            try:
                ship = get_ship(ship_symbol)
                result = getattr(ship, method_name)(arg1)
                return format_result(result)
            except Exception as e:
                return f"Error: {e}"

        ship_method_1arg.__name__ = tool_name
        ship_method_1arg.__doc__ = (
            f"{description}\n\nArgs:\n  ship_symbol: Ship to use\n  arg1: {param_desc}"
        )
        return ship_method_1arg

    elif len(params) == 2:
        param_names = list(params.keys())
        param_descs = list(params.values())

        @tool
        def ship_method_2arg(ship_symbol: str, arg1: str, arg2: int) -> str:
            """Ship method wrapper with 2 args."""
            try:
                ship = get_ship(ship_symbol)
                result = getattr(ship, method_name)(arg1, arg2)
                return format_result(result)
            except Exception as e:
                return f"Error: {e}"

        ship_method_2arg.__name__ = tool_name
        ship_method_2arg.__doc__ = f"{description}\n\nArgs:\n  ship_symbol: Ship to use\n  arg1: {param_descs[0]}\n  arg2: {param_descs[1]}"
        return ship_method_2arg

    elif len(params) == 3:
        param_descs = list(params.values())

        @tool
        def ship_method_3arg(ship_symbol: str, arg1: str, arg2: int, arg3: str) -> str:
            """Ship method wrapper with 3 args."""
            try:
                ship = get_ship(ship_symbol)
                result = getattr(ship, method_name)(arg1, arg2, arg3)
                return format_result(result)
            except Exception as e:
                return f"Error: {e}"

        ship_method_3arg.__name__ = tool_name
        ship_method_3arg.__doc__ = f"{description}\n\nArgs:\n  ship_symbol: Ship to use\n  arg1: {param_descs[0]}\n  arg2: {param_descs[1]}\n  arg3: {param_descs[2]}"
        return ship_method_3arg

    return None


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--api-coverage":
        print_api_coverage()
    else:
        print("Usage:")
        print("  python tool_generator.py --api-coverage    # Show all API endpoints")
        print()
        print("To generate tools in code:")
        print("  from tool_generator import generate_snisp_tools")
        print("  from snisp.agent import Agent")
        print("  agent = Agent(symbol='YOUR_AGENT', token='YOUR_TOKEN')")
        print("  tools = generate_snisp_tools(agent)")
