import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from api_client import SpaceTradersClient

EVENTS_FILE = Path("events.jsonl")
STORY_FILE = Path("story.jsonl")

load_dotenv()
client = SpaceTradersClient(os.environ["TOKEN"])


def _parse_arrival(nav: dict) -> float | None:
    """Return seconds until arrival, or None if not in transit."""
    route = nav.get("route", {})
    arrival_str = route.get("arrival")
    if not arrival_str or nav.get("status") != "IN_TRANSIT":
        return None
    arrival = datetime.fromisoformat(arrival_str.replace("Z", "+00:00"))
    remaining = (arrival - datetime.now(timezone.utc)).total_seconds()
    return max(remaining, 0)


def _wait_with_log(seconds: float, reason: str):
    """Sleep and print periodic status updates."""
    end = time.time() + seconds
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            break
        print(f"  [{reason}] {remaining:.0f}s remaining...")
        time.sleep(min(remaining, 15))


# ──────────────────────────────────────────────
#  Observation tools
# ──────────────────────────────────────────────

@tool
def view_agent() -> str:
    """View your agent's credits, headquarters location, and ship count. Call this first to understand your current situation."""
    data = client.get_agent()
    if "error" in data:
        return f"Error: {data['error']}"
    return (
        f"Agent: {data['symbol']}\n"
        f"Credits: {data['credits']}\n"
        f"Headquarters: {data['headquarters']}\n"
        f"Ship count: {data['shipCount']}\n"
        f"Starting faction: {data['startingFaction']}"
    )


@tool
def view_contracts() -> str:
    """List all your contracts with their status, terms, and delivery requirements. Use this to find contracts to accept and track delivery progress."""
    data = client.list_contracts()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return "No contracts available."
    lines = []
    for c in data:
        lines.append(f"Contract: {c['id']}")
        lines.append(f"  Type: {c['type']}  |  Accepted: {c['accepted']}  |  Fulfilled: {c['fulfilled']}")
        terms = c.get("terms", {})
        lines.append(f"  Payment: {terms.get('payment', {}).get('onAccepted', 0)} on accept, "
                      f"{terms.get('payment', {}).get('onFulfilled', 0)} on fulfill")
        for d in terms.get("deliver", []):
            lines.append(f"  Deliver: {d['unitsRequired']} {d['tradeSymbol']} to {d['destinationSymbol']} "
                          f"({d['unitsFulfilled']}/{d['unitsRequired']} done)")
        lines.append("")
    return "\n".join(lines)


@tool
def view_ships() -> str:
    """List all ships in your fleet with their location, fuel, navigation status, and role. Use this to see what ships you have and where they are."""
    data = client.list_ships()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return "No ships in fleet."
    lines = []
    for s in data:
        nav = s.get("nav", {})
        fuel = s.get("fuel", {})
        lines.append(f"Ship: {s['symbol']}")
        lines.append(f"  Type: {s.get('registration', {}).get('role', '?')}")
        lines.append(f"  Location: {nav.get('waypointSymbol', '?')}  |  Status: {nav.get('status', '?')}")
        lines.append(f"  Fuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}")
        cargo = s.get("cargo", {})
        lines.append(f"  Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', 0)} units")
        lines.append("")
    return "\n".join(lines)


@tool
def view_cargo(ship_symbol: str) -> str:
    """View the cargo contents of a specific ship. Use this to check what ores and goods you're carrying before selling or delivering."""
    data = client.get_cargo(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    inventory = data.get("inventory", [])
    capacity = data.get("capacity", "?")
    units = data.get("units", 0)
    lines = [f"Cargo for {ship_symbol}: {units}/{capacity} units"]
    if not inventory:
        lines.append("  (empty)")
    for item in inventory:
        lines.append(f"  {item['symbol']}: {item['units']} units")
    return "\n".join(lines)


@tool
def find_waypoints(system_symbol: str, trait_or_type: str) -> str:
    """Search for waypoints in a system by trait (e.g. SHIPYARD, MARKETPLACE) or type (e.g. ENGINEERED_ASTEROID). The system_symbol looks like 'X1-AB12'. Use this to find shipyards, asteroids, and markets."""
    # Try as trait first, then as type
    data = client.list_waypoints(system_symbol, traits=trait_or_type)
    if isinstance(data, dict) and "error" in data:
        data = client.list_waypoints(system_symbol, type=trait_or_type)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return f"No waypoints found for '{trait_or_type}' in {system_symbol}."
    lines = []
    for wp in data:
        traits = ", ".join(t["symbol"] for t in wp.get("traits", []))
        lines.append(f"{wp['symbol']} (type: {wp['type']})")
        if traits:
            lines.append(f"  Traits: {traits}")
    return "\n".join(lines)


@tool
def view_shipyard(system_symbol: str, waypoint_symbol: str) -> str:
    """View ships available for purchase at a shipyard. You need a ship present at the waypoint to see prices. Use this before purchasing a ship."""
    data = client.get_shipyard(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    ships = data.get("ships", data.get("shipTypes", []))
    lines = [f"Shipyard at {waypoint_symbol}:"]
    if isinstance(ships, list) and ships:
        for s in ships:
            if isinstance(s, dict) and "name" in s:
                lines.append(f"  {s.get('name', s.get('type', '?'))} — {s.get('purchasePrice', '?')} credits")
                lines.append(f"    Type: {s.get('type', '?')}")
            else:
                lines.append(f"  {s.get('type', str(s))}")
    else:
        lines.append("  No ship details available (need a ship present at this waypoint to see prices).")
    return "\n".join(lines)


@tool
def view_market(system_symbol: str, waypoint_symbol: str) -> str:
    """View market prices at a waypoint. Shows what goods can be bought and sold, and at what prices. You need a ship present to see exact prices."""
    data = client.get_market(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    lines = [f"Market at {waypoint_symbol}:"]
    for section, label in [("exports", "Exports"), ("imports", "Imports"), ("exchange", "Exchange")]:
        items = data.get(section, [])
        if items:
            lines.append(f"  {label}: {', '.join(i['symbol'] for i in items)}")
    trade_goods = data.get("tradeGoods", [])
    if trade_goods:
        lines.append("  Trade goods (with prices):")
        for g in trade_goods:
            lines.append(f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} "
                          f"(volume: {g.get('tradeVolume', '?')})")
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Action tools
# ──────────────────────────────────────────────

@tool
def accept_contract(contract_id: str) -> str:
    """Accept a contract to start working on it. You must accept before you can deliver goods. Accepting often gives you an upfront credit payment."""
    data = client.accept_contract(contract_id)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    agent = data.get("agent", {})
    contract = data.get("contract", {})
    return (
        f"Contract {contract.get('id', contract_id)} accepted!\n"
        f"Credits now: {agent.get('credits', '?')}"
    )


@tool
def purchase_ship(ship_type: str, waypoint_symbol: str) -> str:
    """Purchase a ship at a shipyard waypoint. Common types: SHIP_MINING_DRONE, SHIP_PROBE. You need a ship present at the waypoint and enough credits."""
    data = client.purchase_ship(ship_type, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    agent = data.get("agent", {})
    ship = data.get("ship", {})
    return (
        f"Purchased {ship.get('symbol', '?')} ({ship_type})!\n"
        f"Credits remaining: {agent.get('credits', '?')}"
    )


@tool
def orbit_ship(ship_symbol: str) -> str:
    """Put a ship into orbit at its current waypoint. Ships must be in orbit to navigate or extract ores. Call this after docking or when a ship is docked."""
    data = client.orbit(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    return f"{ship_symbol} is now in orbit at {nav.get('waypointSymbol', '?')}."


@tool
def dock_ship(ship_symbol: str) -> str:
    """Dock a ship at its current waypoint. Ships must be docked to refuel, sell cargo, or purchase ships. Call this before refueling or selling."""
    data = client.dock(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    return f"{ship_symbol} is now docked at {nav.get('waypointSymbol', '?')}."


@tool
def navigate_ship(ship_symbol: str, waypoint_symbol: str) -> str:
    """Navigate a ship to a different waypoint in the same system. The ship must be in orbit first. This consumes fuel and takes time — the tool will wait for arrival automatically."""
    data = client.navigate(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    fuel = data.get("fuel", {})
    wait_secs = _parse_arrival(nav)
    result = f"{ship_symbol} navigating to {waypoint_symbol}."
    result += f"\nFuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}"
    if wait_secs and wait_secs > 0:
        result += f"\nArrival in {wait_secs:.0f}s — waiting..."
        _wait_with_log(wait_secs, f"{ship_symbol} in transit")
        result += f"\n{ship_symbol} has arrived at {waypoint_symbol}."
    else:
        result += f"\n{ship_symbol} has arrived at {waypoint_symbol}."
    return result


@tool
def refuel_ship(ship_symbol: str) -> str:
    """Refuel a ship at the current waypoint. The ship must be docked, and the waypoint must have a marketplace that sells fuel. Do this before long trips."""
    data = client.refuel(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    fuel = data.get("fuel", {})
    tx = data.get("transaction", {})
    return (
        f"{ship_symbol} refueled! Fuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}\n"
        f"Cost: {tx.get('totalPrice', '?')} credits"
    )


@tool
def extract_ore(ship_symbol: str) -> str:
    """Extract ores and minerals from an asteroid. The ship must be in orbit at an asteroid waypoint and have a mining laser. After extraction there is a cooldown — the tool will wait through it automatically."""
    data = client.extract(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    extraction = data.get("extraction", {})
    cooldown = data.get("cooldown", {})
    cargo = data.get("cargo", {})
    yielded = extraction.get("yield", {})
    result = f"Extracted {yielded.get('units', '?')} units of {yielded.get('symbol', '?')}."
    result += f"\nCargo: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    cool_remaining = cooldown.get("remainingSeconds", 0)
    if cool_remaining > 0:
        result += f"\nCooldown: {cool_remaining}s — waiting..."
        _wait_with_log(cool_remaining, f"{ship_symbol} cooldown")
        result += "\nCooldown complete, ready to extract again."
    return result


@tool
def sell_cargo(ship_symbol: str, trade_symbol: str, units: int) -> str:
    """Sell cargo from a ship at the current waypoint's marketplace. The ship must be docked. Specify the trade symbol (e.g. IRON_ORE) and number of units to sell."""
    data = client.sell_cargo(ship_symbol, trade_symbol, units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    tx = data.get("transaction", {})
    cargo = data.get("cargo", {})
    return (
        f"Sold {tx.get('units', units)} {trade_symbol} for {tx.get('totalPrice', '?')} credits.\n"
        f"Cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


@tool
def jettison_cargo(ship_symbol: str, trade_symbol: str, units: int) -> str:
    """Jettison (discard) cargo from a ship. Use this to make room when cargo is full and you don't want to sell. Only jettison non-contract ores."""
    data = client.jettison(ship_symbol, trade_symbol, units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    cargo = data.get("cargo", {})
    return (
        f"Jettisoned {units} {trade_symbol}.\n"
        f"Cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


@tool
def deliver_contract(contract_id: str, ship_symbol: str, trade_symbol: str, units: int) -> str:
    """Deliver goods to fulfill a contract. The ship must be docked at the contract's delivery waypoint and have the required goods in cargo."""
    data = client.deliver_contract(contract_id, ship_symbol, trade_symbol, units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    contract = data.get("contract", {})
    terms = contract.get("terms", {})
    lines = [f"Delivered {units} {trade_symbol} for contract {contract_id}."]
    for d in terms.get("deliver", []):
        lines.append(f"  {d['tradeSymbol']}: {d['unitsFulfilled']}/{d['unitsRequired']} delivered")
    return "\n".join(lines)


@tool
def fulfill_contract(contract_id: str) -> str:
    """Complete a contract after all deliveries are done. This collects the final payment. Only call this when all required goods have been delivered."""
    data = client.fulfill_contract(contract_id)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    agent = data.get("agent", {})
    return (
        f"Contract {contract_id} fulfilled!\n"
        f"Credits now: {agent.get('credits', '?')}"
    )


# ──────────────────────────────────────────────
#  Memory tools
# ──────────────────────────────────────────────

@tool
def recall_memory() -> str:
    """Recall recent events and the narrator's story about your actions. Use this when starting up, after errors, or when you feel stuck in a loop. It shows what you've done recently so you can avoid repeating yourself and make better decisions."""
    lines = []

    # Recent action events (last 15)
    if EVENTS_FILE.exists():
        events = []
        for raw in EVENTS_FILE.read_text(encoding="utf-8").strip().splitlines():
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        recent = [e for e in events if e.get("type") in ("tool_result", "tool_error")][-15:]
        if recent:
            lines.append("=== Recent Actions ===")
            for e in recent:
                tool_name = e.get("tool", "?")
                if e.get("type") == "tool_error":
                    lines.append(f"  FAILED {tool_name}: {e.get('error', '')[:200]}")
                else:
                    lines.append(f"  {tool_name}: {e.get('result', '')[:200]}")

    # Story segments (last 5)
    if STORY_FILE.exists():
        segments = []
        for raw in STORY_FILE.read_text(encoding="utf-8").strip().splitlines():
            try:
                segments.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        recent_story = segments[-5:]
        if recent_story:
            lines.append("\n=== The Story So Far ===")
            for s in recent_story:
                lines.append(f"  [{s.get('tool', '?')}] {s.get('story', '')[:300]}")

    if not lines:
        return "No memories yet — this is a fresh start. Check your agent status, ships, and contracts to get oriented."

    return "\n".join(lines)


ALL_TOOLS = [
    view_agent, view_contracts, view_ships, view_cargo,
    find_waypoints, view_shipyard, view_market,
    accept_contract, purchase_ship, orbit_ship, dock_ship,
    navigate_ship, refuel_ship, extract_ore, sell_cargo,
    jettison_cargo, deliver_contract, fulfill_contract,
    recall_memory,
]
