import os
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.tools import tool

from api_client import SpaceTradersClient

load_dotenv()
client = SpaceTradersClient(os.environ["TOKEN"])


@dataclass
class ToolResult:
    """Result from a tool execution, with optional wait time for async narrative."""
    message: str
    wait_seconds: float = 0.0
    tool_name: str = ""
    is_significant: bool = False  # True for action tools worth narrating


def _parse_arrival(nav: dict) -> float:
    """Return seconds until arrival, or 0 if not in transit."""
    route = nav.get("route", {})
    arrival_str = route.get("arrival")
    if not arrival_str or nav.get("status") != "IN_TRANSIT":
        return 0.0
    arrival = datetime.fromisoformat(arrival_str.replace("Z", "+00:00"))
    remaining = (arrival - datetime.now(timezone.utc)).total_seconds()
    return max(remaining, 0.0)


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


def _get_ship_capabilities(ship: dict) -> list[str]:
    """Extract capability tags from ship mounts and modules."""
    caps = []
    mounts = ship.get("mounts", [])
    modules = ship.get("modules", [])

    mount_symbols = [m.get("symbol", "") for m in mounts]
    module_symbols = [m.get("symbol", "") for m in modules]

    if any("MINING" in s or "LASER" in s for s in mount_symbols):
        caps.append("CAN_MINE")
    if any("SURVEY" in s for s in mount_symbols):
        caps.append("CAN_SURVEY")
    if any("SIPHON" in s for s in mount_symbols):
        caps.append("CAN_SIPHON")
    if any("SENSOR" in s for s in mount_symbols):
        caps.append("CAN_SCAN")
    if any("REFINERY" in s or "PROCESSOR" in s for s in module_symbols):
        caps.append("CAN_REFINE")
    if any("WARP" in s for s in modules):
        caps.append("CAN_WARP")

    return caps


@tool
def view_ships() -> str:
    """List all ships in your fleet with their location, fuel, navigation status, role, and capabilities. Use this to see what ships you have and what they can do."""
    data = client.list_ships()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return "No ships in fleet."
    lines = []
    for s in data:
        nav = s.get("nav", {})
        fuel = s.get("fuel", {})
        cargo = s.get("cargo", {})
        caps = _get_ship_capabilities(s)

        lines.append(f"Ship: {s['symbol']}")
        lines.append(f"  Role: {s.get('registration', {}).get('role', '?')}")
        lines.append(f"  Location: {nav.get('waypointSymbol', '?')}  |  Status: {nav.get('status', '?')}")
        lines.append(f"  Fuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}")
        lines.append(f"  Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', 0)} units")
        if caps:
            lines.append(f"  Capabilities: {', '.join(caps)}")
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
def view_ship_details(ship_symbol: str) -> str:
    """View detailed ship information including mounts (weapons, mining lasers, etc.) and modules (cargo, refineries, etc.). Use this to understand what a ship can do."""
    ships = client.list_ships()
    if isinstance(ships, dict) and "error" in ships:
        return f"Error: {ships['error']}"

    ship = None
    for s in ships:
        if s.get("symbol") == ship_symbol:
            ship = s
            break

    if not ship:
        return f"Error: Ship {ship_symbol} not found"

    lines = [f"=== {ship_symbol} Details ==="]

    # Basic info
    reg = ship.get("registration", {})
    lines.append(f"Role: {reg.get('role', '?')}")
    lines.append(f"Faction: {reg.get('factionSymbol', '?')}")

    # Frame
    frame = ship.get("frame", {})
    lines.append(f"\nFrame: {frame.get('name', '?')} ({frame.get('symbol', '?')})")
    lines.append(f"  Module slots: {frame.get('moduleSlots', '?')}, Mounting points: {frame.get('mountingPoints', '?')}")

    # Engine
    engine = ship.get("engine", {})
    lines.append(f"\nEngine: {engine.get('name', '?')} (speed: {engine.get('speed', '?')})")

    # Mounts (weapons, lasers, etc.)
    mounts = ship.get("mounts", [])
    if mounts:
        lines.append(f"\nMounts ({len(mounts)}):")
        for m in mounts:
            lines.append(f"  • {m.get('name', m.get('symbol', '?'))}")
            if m.get('strength'):
                lines.append(f"    Strength: {m.get('strength')}")
    else:
        lines.append("\nMounts: None")

    # Modules
    modules = ship.get("modules", [])
    if modules:
        lines.append(f"\nModules ({len(modules)}):")
        for m in modules:
            cap = f" (capacity: {m.get('capacity')})" if m.get('capacity') else ""
            lines.append(f"  • {m.get('name', m.get('symbol', '?'))}{cap}")
    else:
        lines.append("\nModules: None")

    # Capabilities summary
    caps = _get_ship_capabilities(ship)
    if caps:
        lines.append(f"\nCapabilities: {', '.join(caps)}")

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
    """Navigate a ship to a different waypoint in the same system. The ship must be in orbit first. This consumes fuel and takes time."""
    data = client.navigate(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    fuel = data.get("fuel", {})
    wait_secs = _parse_arrival(nav)
    result = f"{ship_symbol} navigating to {waypoint_symbol}."
    result += f"\nFuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}"
    if wait_secs > 0:
        result += f"\nArrival in {wait_secs:.0f}s."
    else:
        result += f"\n{ship_symbol} has arrived at {waypoint_symbol}."
    # Store wait time in a way the agent loop can access
    navigate_ship._last_wait = wait_secs
    return result

# Initialize the wait tracking attribute
navigate_ship._last_wait = 0.0


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
    """Extract ores and minerals from an asteroid. The ship must be in orbit at an asteroid waypoint and have a mining laser. After extraction there is a cooldown before you can extract again."""
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
        result += f"\nCooldown: {cool_remaining}s before next extraction."
    # Store wait time in a way the agent loop can access
    extract_ore._last_wait = float(cool_remaining)
    return result

# Initialize the wait tracking attribute
extract_ore._last_wait = 0.0


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
def transfer_cargo(from_ship: str, to_ship: str, trade_symbol: str, units: int) -> str:
    """Transfer cargo between two ships. Both ships must be at the same waypoint and in ORBIT. Use this to offload cargo from a miner to a larger cargo ship."""
    data = client.transfer_cargo(from_ship, to_ship, trade_symbol, units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    cargo = data.get("cargo", {})
    return (
        f"Transferred {units} {trade_symbol} from {from_ship} to {to_ship}.\n"
        f"{from_ship} cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


# ──────────────────────────────────────────────
#  Advanced ship operations
# ──────────────────────────────────────────────

@tool
def survey_asteroid(ship_symbol: str) -> str:
    """Survey an asteroid field to find rich deposits. The ship must be in orbit at an asteroid. Surveys improve extraction yields but expire after a few minutes. Returns survey data to use with extract_with_survey."""
    data = client.survey(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    surveys = data.get("surveys", [])
    cooldown = data.get("cooldown", {})
    lines = [f"Created {len(surveys)} survey(s). Cooldown: {cooldown.get('remainingSeconds', 0)}s"]
    for i, survey in enumerate(surveys):
        deposits = [d.get("symbol", "?") for d in survey.get("deposits", [])]
        lines.append(f"  Survey {i+1}: {survey.get('signature', '?')} - Size: {survey.get('size', '?')}")
        lines.append(f"    Deposits: {', '.join(deposits)}")
        lines.append(f"    Expires: {survey.get('expiration', '?')}")
    return "\n".join(lines)


@tool
def scan_waypoints(ship_symbol: str) -> str:
    """Scan for waypoints from current location. Reveals nearby waypoints in the system. Ship must be in orbit."""
    data = client.scan_waypoints(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    waypoints = data.get("waypoints", [])
    cooldown = data.get("cooldown", {})
    lines = [f"Found {len(waypoints)} waypoints. Cooldown: {cooldown.get('remainingSeconds', 0)}s"]
    for wp in waypoints[:15]:  # Limit output
        traits = ", ".join(t.get("symbol", "") for t in wp.get("traits", [])[:3])
        lines.append(f"  {wp.get('symbol', '?')} ({wp.get('type', '?')}) - {traits}")
    if len(waypoints) > 15:
        lines.append(f"  ... and {len(waypoints) - 15} more")
    return "\n".join(lines)


@tool
def scan_ships(ship_symbol: str) -> str:
    """Scan for other ships in the area. Ship must be in orbit."""
    data = client.scan_ships(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    ships = data.get("ships", [])
    cooldown = data.get("cooldown", {})
    lines = [f"Found {len(ships)} ships. Cooldown: {cooldown.get('remainingSeconds', 0)}s"]
    for ship in ships[:10]:
        nav = ship.get("nav", {})
        lines.append(f"  {ship.get('symbol', '?')} - {ship.get('registration', {}).get('role', '?')} @ {nav.get('waypointSymbol', '?')}")
    return "\n".join(lines)


@tool
def jump_ship(ship_symbol: str, system_symbol: str) -> str:
    """Jump ship to another star system through a jump gate. Ship must be at a jump gate waypoint and in orbit. Requires antimatter fuel."""
    data = client.jump(ship_symbol, system_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    cooldown = data.get("cooldown", {})
    return (
        f"{ship_symbol} jumped to system {system_symbol}.\n"
        f"Now at: {nav.get('waypointSymbol', '?')}\n"
        f"Cooldown: {cooldown.get('remainingSeconds', 0)}s"
    )


@tool
def warp_ship(ship_symbol: str, waypoint_symbol: str) -> str:
    """Warp ship to a waypoint in another system. Uses antimatter. Ship must be in orbit."""
    data = client.warp(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    fuel = data.get("fuel", {})
    return (
        f"{ship_symbol} warping to {waypoint_symbol}.\n"
        f"Fuel: {fuel.get('current', '?')}/{fuel.get('capacity', '?')}"
    )


@tool
def negotiate_contract(ship_symbol: str) -> str:
    """Negotiate a new contract at a faction headquarters. Ship must be docked at a faction HQ waypoint."""
    data = client.negotiate_contract(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    contract = data.get("contract", {})
    terms = contract.get("terms", {})
    lines = [f"Negotiated new contract: {contract.get('id', '?')}"]
    lines.append(f"  Type: {contract.get('type', '?')}")
    lines.append(f"  Payment: {terms.get('payment', {}).get('onAccepted', 0)} on accept, "
                 f"{terms.get('payment', {}).get('onFulfilled', 0)} on fulfill")
    for d in terms.get("deliver", []):
        lines.append(f"  Deliver: {d.get('unitsRequired', '?')} {d.get('tradeSymbol', '?')} to {d.get('destinationSymbol', '?')}")
    return "\n".join(lines)


@tool
def chart_waypoint(ship_symbol: str) -> str:
    """Chart the current waypoint to add it to your known waypoints. Useful for unexplored locations."""
    data = client.chart(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    chart = data.get("chart", {})
    waypoint = data.get("waypoint", {})
    return (
        f"Charted waypoint: {waypoint.get('symbol', '?')}\n"
        f"Type: {waypoint.get('type', '?')}\n"
        f"Submitted by: {chart.get('submittedBy', '?')}"
    )


@tool
def view_jump_gate(system_symbol: str, waypoint_symbol: str) -> str:
    """View jump gate connections from a waypoint. Shows which systems are reachable."""
    data = client.get_jump_gate(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    connections = data.get("connections", [])
    lines = [f"Jump gate at {waypoint_symbol} connects to {len(connections)} systems:"]
    for conn in connections[:20]:
        lines.append(f"  → {conn}")
    if len(connections) > 20:
        lines.append(f"  ... and {len(connections) - 20} more")
    return "\n".join(lines)


@tool
def set_flight_mode(ship_symbol: str, mode: str) -> str:
    """Set ship flight mode. Modes: CRUISE (balanced), BURN (fast, uses more fuel), DRIFT (slow, no fuel), STEALTH (avoid detection)."""
    data = client.set_flight_mode(ship_symbol, mode.upper())
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    return f"{ship_symbol} flight mode set to {mode.upper()}"


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
#  Wait tool
# ──────────────────────────────────────────────

@tool
def wait(seconds: int) -> str:
    """Wait for a specified number of seconds. Use this when all your ships are on cooldown or in transit and there's nothing else to do. Maximum wait is 300 seconds (5 minutes)."""
    import time
    # Cap at 5 minutes
    seconds = min(seconds, 300)
    if seconds <= 0:
        return "No waiting needed."

    time.sleep(seconds)
    return f"Waited {seconds} seconds. Ships may now be available."

# Track wait time for the bot loop
wait._last_wait = 0.0


# ──────────────────────────────────────────────
#  Tool registry
# ──────────────────────────────────────────────

# All tools available for the agent
ALL_TOOLS = [
    # Observation
    view_agent, view_contracts, view_ships, view_ship_details, view_cargo,
    find_waypoints, view_shipyard, view_market, view_jump_gate,
    # Basic ship operations
    orbit_ship, dock_ship, navigate_ship, refuel_ship,
    # Mining & resources
    extract_ore, survey_asteroid,
    # Trading & cargo
    sell_cargo, jettison_cargo, transfer_cargo,
    # Contracts
    accept_contract, deliver_contract, fulfill_contract, negotiate_contract,
    # Ships & shipyard
    purchase_ship,
    # Scanning & exploration
    scan_waypoints, scan_ships, chart_waypoint,
    # Inter-system travel
    jump_ship, warp_ship, set_flight_mode,
    # Note: wait tool removed - there's always something productive to do
]

# Tools that are "significant" actions worth narrating
SIGNIFICANT_TOOLS = {
    "extract_ore", "survey_asteroid", "navigate_ship", "sell_cargo", "jettison_cargo",
    "transfer_cargo", "accept_contract", "fulfill_contract", "negotiate_contract",
    "purchase_ship", "refuel_ship", "dock_ship", "orbit_ship", "deliver_contract",
    "scan_waypoints", "scan_ships", "chart_waypoint", "jump_ship", "warp_ship",
}

# Tools that have wait/cooldown times (for parallel narrative)
WAITING_TOOLS = {"navigate_ship", "extract_ore", "survey_asteroid", "scan_waypoints", "scan_ships", "jump_ship", "warp_ship"}


def get_tool_by_name(name: str):
    """Get a tool function by its name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None


def get_last_wait(tool_name: str) -> float:
    """Get the last wait time from a waiting tool."""
    if tool_name == "navigate_ship":
        return getattr(navigate_ship, "_last_wait", 0.0)
    elif tool_name == "extract_ore":
        return getattr(extract_ore, "_last_wait", 0.0)
    return 0.0
