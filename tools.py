import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

from api_client import SpaceTradersClient

load_dotenv()
client = SpaceTradersClient(os.environ["TOKEN"])


MARKET_CACHE_FILE = Path("market_cache.json")


def load_market_cache() -> dict:
    """Load cached market data."""
    if MARKET_CACHE_FILE.exists():
        try:
            return json.loads(MARKET_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_market_to_cache(waypoint_symbol: str, data: dict):
    """Save market data to cache after a view_market call.

    Structural data (imports/exports/exchange) is stable and saved even without ship in orbit.
    Price data (tradeGoods) requires ship in orbit and is marked with timestamp.
    """
    import time
    cache = load_market_cache()

    # Get existing entry or create new one
    entry = cache.get(waypoint_symbol, {})

    # Structural data (stable - what market can buy/sell)
    # Always update this if available
    for section in ("exports", "imports", "exchange"):
        items = data.get(section, [])
        if items:
            entry[section] = [i["symbol"] for i in items]

    # Price data (dynamic - requires ship in orbit)
    trade_goods = data.get("tradeGoods", [])
    if trade_goods:
        entry["trade_goods"] = [
            {
                "symbol": g["symbol"],
                "purchasePrice": g.get("purchasePrice"),  # What market PAYS for items
                "sellPrice": g.get("sellPrice"),          # What market SELLS items for
                "tradeVolume": g.get("tradeVolume")
            }
            for g in trade_goods
        ]
        # Mark when we got price data
        entry["last_updated"] = int(time.time())

    # Save even if we only got structural data (no prices)
    if entry:
        cache[waypoint_symbol] = entry
        MARKET_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _ensure_orbit(ship_symbol: str) -> str | None:
    """Ensure ship is in orbit. Auto-transitions from DOCKED. Returns error string or None."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Ship {ship_symbol} not found: {ship['error']}"
    status = ship.get("nav", {}).get("status", "")
    if status == "IN_ORBIT":
        return None
    if status == "DOCKED":
        result = client.orbit(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return f"Could not orbit {ship_symbol}: {result['error']}"
        return None
    if status == "IN_TRANSIT":
        return f"{ship_symbol} is currently in transit and cannot change state"
    return None


def _ensure_dock(ship_symbol: str) -> str | None:
    """Ensure ship is docked. Auto-transitions from IN_ORBIT. Returns error string or None."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Ship {ship_symbol} not found: {ship['error']}"
    status = ship.get("nav", {}).get("status", "")
    if status == "DOCKED":
        return None
    if status == "IN_ORBIT":
        result = client.dock(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return f"Could not dock {ship_symbol}: {result['error']}"
        return None
    if status == "IN_TRANSIT":
        return f"{ship_symbol} is currently in transit and cannot change state"
    return None


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

        role = s.get('registration', {}).get('role', '?')
        lines.append(f"Ship: {s['symbol']}")
        lines.append(f"  Role: {role}")
        lines.append(f"  Location: {nav.get('waypointSymbol', '?')}  |  Status: {nav.get('status', '?')}")

        # Make it VERY clear when ship doesn't use fuel
        if fuel.get('capacity', 0) == 0:
            lines.append(f"  Fuel: SOLAR POWERED (FREE MOVEMENT)")
        else:
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
def find_waypoints(system_symbol: str, trait_or_type: str, reference_ship: str = None) -> str:
    """Search for waypoints in a system by TRAIT (e.g. SHIPYARD, MARKETPLACE, COMMON_METAL_DEPOSITS) or TYPE (e.g. ASTEROID, ENGINEERED_ASTEROID, PLANET, GAS_GIANT).

    IMPORTANT: Specify reference_ship when planning that ship's actions!
    - With reference_ship: Results sorted by distance from THAT SPECIFIC SHIP (closest first)
    - Without reference_ship: Results in arbitrary order (no distance shown)

    Example: Planning where WHATER-3 should mine? Use find_waypoints('X1-AB12', 'ASTEROID', reference_ship='WHATER-3')
    This prevents confusion when satellite is at location but cargo ship is far away.

    IMPORTANT: This searches by waypoint TYPE/TRAIT, NOT by resource. You CANNOT search for 'ALUMINUM_ORE' or 'IRON_ORE' asteroids.
    To find mineable asteroids, search for type='ASTEROID' or 'ENGINEERED_ASTEROID', then extract to see what resources they produce.

    The system_symbol looks like 'X1-AB12'."""
    import math

    # Try as trait first, then as type
    data = client.list_waypoints(system_symbol, traits=trait_or_type)
    if isinstance(data, dict) and "error" in data:
        data = client.list_waypoints(system_symbol, type=trait_or_type)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    # MAGIC: When searching for ASTEROID, also include ENGINEERED_ASTEROID (starter asteroids)
    if trait_or_type.upper() == "ASTEROID" and isinstance(data, list):
        engineered = client.list_waypoints(system_symbol, type="ENGINEERED_ASTEROID")
        if isinstance(engineered, list):
            data.extend(engineered)

    if not data:
        return f"No waypoints found for '{trait_or_type}' in {system_symbol}."

    # Get reference ship position if specified
    reference_position = None
    reference_ship_name = None

    if reference_ship:
        ship = client.get_ship(reference_ship)
        if isinstance(ship, dict) and "error" not in ship:
            nav = ship.get("nav", {})
            current_wp = nav.get("waypointSymbol", "")

            # Find reference ship's coordinates
            waypoints = client.list_waypoints(system_symbol)
            if isinstance(waypoints, list):
                for wp in waypoints:
                    if wp.get("symbol") == current_wp:
                        reference_position = (wp.get("x", 0), wp.get("y", 0))
                        reference_ship_name = reference_ship
                        break

    # Calculate distance from reference point
    def calculate_distance(waypoint):
        if not reference_position:
            return 0  # No reference for sorting
        wx, wy = waypoint.get("x", 0), waypoint.get("y", 0)
        rx, ry = reference_position
        return math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)

    # Sort waypoints by distance if we have a reference
    if reference_position:
        sorted_waypoints = sorted(data, key=calculate_distance)
        header = f"Waypoints sorted by distance from {reference_ship_name}:\n"
    else:
        sorted_waypoints = data
        header = "Waypoints (no reference ship specified - distances not shown):\n"

    lines = [header]
    for wp in sorted_waypoints:
        traits = ", ".join(t["symbol"] for t in wp.get("traits", []))

        # Show distance if we have a reference position
        if reference_position:
            dist = calculate_distance(wp)
            lines.append(f"{wp['symbol']} (type: {wp['type']}, distance from {reference_ship_name}: {dist:.1f})")
        else:
            lines.append(f"{wp['symbol']} (type: {wp['type']})")

        if traits:
            lines.append(f"  Traits: {traits}")

    return "\n".join(lines)


@tool
def scan_system(system_symbol: str, reference_ship: str = None, closest_only: bool = False, within_cruise_range: bool = False) -> str:
    """Scan all waypoints in a system to discover markets, shipyards, and resources WITHOUT requiring ship visits.

    This is VERY efficient - one API call reveals structural market data (what each market imports/exports) for the entire system.
    Use this early to understand the system's economy before moving ships around.

    Args:
        system_symbol: System to scan (e.g., 'X1-AB12') or waypoint (e.g., 'X1-AB12-C3') - will extract system automatically
        reference_ship: Ship symbol to calculate distances from (e.g., 'WHATER-1')
                       If provided, results are ALWAYS sorted by distance (closest first)
        closest_only: If True, return only the closest waypoint of each category (1 market, 1 shipyard, 1 asteroid)
        within_cruise_range: If True and reference_ship provided, filter to only waypoints within ship's CRUISE fuel range

    The system_symbol looks like 'X1-AB12'. If you pass a waypoint like 'X1-AB12-C3', the system will be extracted."""
    import math

    # Extract system from waypoint if needed (e.g., 'X1-KD26-A1' -> 'X1-KD26')
    # System format: SECTOR-SYSTEM (e.g., X1-KD26)
    # Waypoint format: SECTOR-SYSTEM-WAYPOINT (e.g., X1-KD26-A1)
    parts = system_symbol.split('-')
    if len(parts) > 2:
        # This is a waypoint, extract system (first two parts)
        system_symbol = f"{parts[0]}-{parts[1]}"

    # Get all waypoints in the system
    waypoints = client.list_waypoints(system_symbol)
    if isinstance(waypoints, dict) and "error" in waypoints:
        return f"Error: {waypoints['error']}"
    if not waypoints:
        return f"No waypoints found in system {system_symbol}."

    # Get reference ship position and fuel if specified
    reference_position = None
    reference_ship_name = None
    ship_fuel_capacity = 0

    if reference_ship:
        ship = client.get_ship(reference_ship)
        if isinstance(ship, dict) and "error" not in ship:
            nav = ship.get("nav", {})
            fuel = ship.get("fuel", {})
            current_wp = nav.get("waypointSymbol", "")
            ship_fuel_capacity = fuel.get("capacity", 0)

            # Find reference ship's coordinates
            for wp in waypoints:
                if wp.get("symbol") == current_wp:
                    reference_position = (wp.get("x", 0), wp.get("y", 0))
                    reference_ship_name = reference_ship
                    break

    # Helper function to calculate distance
    def calc_distance(wp):
        if not reference_position:
            return 0
        wx, wy = wp.get("x", 0), wp.get("y", 0)
        rx, ry = reference_position
        return math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)

    # Process waypoints and extract market data
    markets_found = 0
    shipyards_found = 0
    asteroids_found = 0

    lines = [f"System Scan: {system_symbol} ({len(waypoints)} waypoints)\n"]

    # Categorize and display waypoints
    market_waypoints = []
    shipyard_waypoints = []
    asteroid_waypoints = []
    other_waypoints = []

    for wp in waypoints:
        traits = [t.get("symbol", "") for t in wp.get("traits", [])]

        if "MARKETPLACE" in traits:
            market_waypoints.append(wp)
            markets_found += 1

            # Cache structural market data if available
            # SpaceTraders API includes imports/exports in waypoint data
            cache = load_market_cache()
            wp_symbol = wp.get("symbol")

            # Check if waypoint has market trade data
            if wp_symbol and wp_symbol not in cache:
                # Extract imports/exports from traits or modifiers
                # The API structure varies, check for 'imports', 'exports', 'exchange'
                entry = {}

                # Try to find market data in the waypoint object
                for key in ["imports", "exports", "exchange"]:
                    if key in wp and wp[key]:
                        items = wp[key]
                        if isinstance(items, list):
                            entry[key] = [i.get("symbol") if isinstance(i, dict) else str(i) for i in items]

                # Save to cache if we found any market data
                if entry:
                    cache[wp_symbol] = entry
                    MARKET_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        if "SHIPYARD" in traits:
            shipyard_waypoints.append(wp)
            shipyards_found += 1

        wp_type = wp.get("type", "")
        if "ASTEROID" in wp_type:
            asteroid_waypoints.append(wp)
            asteroids_found += 1
        elif "MARKETPLACE" not in traits and "SHIPYARD" not in traits:
            other_waypoints.append(wp)

    # Sort by distance if reference ship specified
    if reference_position:
        market_waypoints.sort(key=calc_distance)
        shipyard_waypoints.sort(key=calc_distance)
        asteroid_waypoints.sort(key=calc_distance)

    # Filter by cruise range if requested
    if within_cruise_range and reference_position and ship_fuel_capacity > 0:
        max_distance = ship_fuel_capacity  # CRUISE uses ~1 fuel per distance
        market_waypoints = [wp for wp in market_waypoints if calc_distance(wp) <= max_distance]
        shipyard_waypoints = [wp for wp in shipyard_waypoints if calc_distance(wp) <= max_distance]
        asteroid_waypoints = [wp for wp in asteroid_waypoints if calc_distance(wp) <= max_distance]
        lines.append(f"Filtered to within CRUISE range ({max_distance} units):\n")

    # Limit to closest only if requested
    if closest_only:
        market_waypoints = market_waypoints[:1]
        shipyard_waypoints = shipyard_waypoints[:1]
        asteroid_waypoints = asteroid_waypoints[:1]

    # Summary
    if reference_ship_name:
        lines.append(f"System Scan (distances from {reference_ship_name}):")
    lines.append(f"Markets: {len(market_waypoints)} | Shipyards: {len(shipyard_waypoints)} | Asteroids: {len(asteroid_waypoints)}\n")

    # Show markets with structural data
    if market_waypoints:
        lines.append("=== MARKETS ===")
        for wp in market_waypoints:
            wp_symbol = wp.get("symbol")
            traits = [t.get("symbol", "") for t in wp.get("traits", [])]

            # Calculate distance if we have reference position
            dist_str = ""
            if reference_position:
                dist = calc_distance(wp)
                dist_str = f" (distance from {reference_ship_name}: {dist:.1f})"

            lines.append(f"\n{wp_symbol}{dist_str}")
            lines.append(f"  Type: {wp.get('type', '?')}")

            # Show imports/exports if available
            if "imports" in wp:
                imports = [i.get("symbol") if isinstance(i, dict) else str(i) for i in wp.get("imports", [])]
                if imports:
                    lines.append(f"  Imports (buys): {', '.join(imports[:10])}")
            if "exports" in wp:
                exports = [e.get("symbol") if isinstance(e, dict) else str(e) for e in wp.get("exports", [])]
                if exports:
                    lines.append(f"  Exports (sells): {', '.join(exports[:10])}")
            if "exchange" in wp:
                exchange = [e.get("symbol") if isinstance(e, dict) else str(e) for e in wp.get("exchange", [])]
                if exchange:
                    lines.append(f"  Exchange: {', '.join(exchange)}")

    # Show shipyards
    if shipyard_waypoints:
        lines.append("\n=== SHIPYARDS ===")
        display_shipyards = shipyard_waypoints[:5] if not closest_only else shipyard_waypoints
        for wp in display_shipyards:
            dist_str = ""
            if reference_position:
                dist = calc_distance(wp)
                dist_str = f" (distance from {reference_ship_name}: {dist:.1f})"
            lines.append(f"{wp.get('symbol')}{dist_str}")

    # Show asteroids (brief)
    if asteroid_waypoints:
        lines.append(f"\n=== ASTEROIDS ===")
        # Already sorted by distance if reference_position exists
        # Show first 5 (or all if closest_only was used)
        display_asteroids = asteroid_waypoints[:5] if not closest_only else asteroid_waypoints

        for wp in display_asteroids:
            dist_str = ""
            if reference_position:
                dist = calc_distance(wp)
                dist_str = f" (distance from {reference_ship_name}: {dist:.1f})"

            traits = [t.get("symbol", "") for t in wp.get("traits", []) if t.get("symbol") not in ["MARKETPLACE", "SHIPYARD"]]
            trait_str = f" - {', '.join(traits)}" if traits else ""
            lines.append(f"{wp.get('symbol')} ({wp.get('type')}){dist_str}{trait_str}")

    return "\n".join(lines)


@tool
def view_shipyard(waypoint_symbol: str) -> str:
    """View ships available for purchase at a shipyard. You need a ship present at the waypoint to see prices. Use this before purchasing a ship.

    Args:
        waypoint_symbol: Waypoint with shipyard (e.g., 'X1-KD26-A1')
    """
    # Extract system from waypoint (e.g., 'X1-KD26-A1' -> 'X1-KD26')
    system_symbol = '-'.join(waypoint_symbol.split('-')[:2])

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
def view_market(waypoint_symbol: str) -> str:
    """View market prices and shipyard info at a waypoint. Shows trade goods, imports/exports, and ships for sale if a shipyard is present. You need a ship present to see exact prices.

    Args:
        waypoint_symbol: Waypoint with market (e.g., 'X1-KD26-B7')
    """
    # Extract system from waypoint (e.g., 'X1-KD26-B7' -> 'X1-KD26')
    system_symbol = '-'.join(waypoint_symbol.split('-')[:2])

    lines = []

    # Market data
    try:
        data = client.get_market(system_symbol, waypoint_symbol)
    except Exception as e:
        return f"Error calling get_market: {str(e)}"

    if data is None:
        lines.append(f"API returned None for market at {waypoint_symbol}")
    elif isinstance(data, dict) and "error" in data:
        lines.append(f"Error getting market at {waypoint_symbol}: {data['error']}")
    elif isinstance(data, dict):
        # Check if data is completely empty
        if not data:
            lines.append(f"Market API returned empty dict for {waypoint_symbol}")
        else:
            try:
                lines.append(f"Market at {waypoint_symbol}:")
                has_data = False
                for section, label in [("exports", "Exports"), ("imports", "Imports"), ("exchange", "Exchange")]:
                    items = data.get(section, [])
                    if items:
                        has_data = True
                        lines.append(f"  {label}: {', '.join(i['symbol'] for i in items)}")
                trade_goods = data.get("tradeGoods", [])
                if trade_goods:
                    has_data = True
                    lines.append("  Trade goods (with prices):")
                    for g in trade_goods:
                        lines.append(f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} "
                                      f"(volume: {g.get('tradeVolume', '?')})")

                if not has_data:
                    lines.append(f"  (No market data available - may need ship in orbit to see prices)")

                # Cache market data for future reference
                _save_market_to_cache(waypoint_symbol, data)
            except Exception as e:
                lines.append(f"Error processing market data: {str(e)}")
                lines.append(f"Raw data keys: {list(data.keys())}")
    else:
        lines.append(f"Unexpected data type from API: {type(data)}")

    # Shipyard data (if present at same waypoint)
    try:
        shipyard = client.get_shipyard(system_symbol, waypoint_symbol)
        if isinstance(shipyard, dict) and "error" not in shipyard and shipyard:
            ships = shipyard.get("ships", shipyard.get("shipTypes", []))
            lines.append(f"\nShipyard at {waypoint_symbol}:")
            if isinstance(ships, list) and ships:
                for s in ships:
                    if isinstance(s, dict) and "name" in s:
                        lines.append(f"  {s.get('name', s.get('type', '?'))} — {s.get('purchasePrice', '?')} credits")
                    else:
                        lines.append(f"  {s.get('type', str(s))}")
            else:
                lines.append("  No ship details available (need a ship present to see prices).")
    except Exception:
        pass  # Shipyard not present, that's OK

    if not lines:
        return f"No data available for {waypoint_symbol} (API returned nothing)"

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
def navigate_ship(ship_symbol: str, waypoint_symbol: str, use_drift: bool = False) -> str:
    """Navigate a ship to a different waypoint in the same system. Auto-orbits if docked. Consumes fuel and takes time.

    IMPORTANT: If ship doesn't have enough fuel, this will return an ERROR instead of automatically using DRIFT mode.
    DRIFT mode is 10x slower! You must explicitly set use_drift=True to confirm you want slow DRIFT navigation.

    Args:
        ship_symbol: The ship to navigate
        waypoint_symbol: The destination waypoint
        use_drift: Set to True to explicitly confirm DRIFT mode (slow but free). Default False.
    """
    import math

    # Check fuel BEFORE attempting navigation
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"

    nav = ship.get("nav", {})
    fuel = ship.get("fuel", {})
    current_wp = nav.get("waypointSymbol", "")
    system = nav.get("systemSymbol", "")
    fuel_current = fuel.get("current", 0)
    fuel_capacity = fuel.get("capacity", 0)

    # Solar powered ships don't need fuel checks
    if fuel_capacity > 0:
        # Calculate fuel requirement
        waypoints = client.list_waypoints(system)
        if isinstance(waypoints, list):
            origin = None
            dest = None
            for wp in waypoints:
                if wp.get("symbol") == current_wp:
                    origin = wp
                if wp.get("symbol") == waypoint_symbol:
                    dest = wp

            if origin and dest:
                dx = dest.get("x", 0) - origin.get("x", 0)
                dy = dest.get("y", 0) - origin.get("y", 0)
                distance = math.sqrt(dx * dx + dy * dy)
                est_fuel = max(1, round(distance))

                # Check if we have enough fuel
                if fuel_current < est_fuel:
                    # Not enough fuel! Check if we're at a fuel station
                    market_cache = load_market_cache()
                    current_market = market_cache.get(current_wp, {})
                    has_fuel = "FUEL" in current_market.get("exchange", [])

                    if has_fuel:
                        return (f"Error: Not enough fuel! {ship_symbol} has {fuel_current}/{fuel_capacity} fuel "
                                f"but needs ~{est_fuel} for this trip.\n"
                                f"You are at {current_wp} which sells FUEL. Call refuel_ship('{ship_symbol}') first!")

                    # Not at fuel station - require explicit DRIFT confirmation
                    if not use_drift:
                        return (f"Error: Not enough fuel! {ship_symbol} has {fuel_current}/{fuel_capacity} fuel "
                                f"but needs ~{est_fuel} for this trip.\n"
                                f"DRIFT mode available but is 10x SLOWER (~{int(distance * 10)}s instead of ~{int(distance)}s).\n"
                                f"Options:\n"
                                f"1. RECOMMENDED: Find and refuel at a market with FUEL (check scan_system or Known Markets)\n"
                                f"2. Use DRIFT: Call navigate_ship('{ship_symbol}', '{waypoint_symbol}', use_drift=True)")

    # Fuel check passed or DRIFT explicitly confirmed - proceed with navigation
    err = _ensure_orbit(ship_symbol)
    if err:
        return f"Error: {err}"

    # BUG FIX: Set the correct flight mode before navigation
    current_flight_mode = nav.get("flightMode", "CRUISE")

    if use_drift:
        # Explicitly requesting DRIFT mode - set it if not already in DRIFT
        if current_flight_mode != "DRIFT":
            result = client.set_flight_mode(ship_symbol, "DRIFT")
            if isinstance(result, dict) and "error" in result:
                return f"Error switching to DRIFT mode: {result['error']}"
    else:
        # NOT using DRIFT - make sure we're in CRUISE if ship is stuck in DRIFT
        if current_flight_mode == "DRIFT":
            result = client.set_flight_mode(ship_symbol, "CRUISE")
            if isinstance(result, dict) and "error" in result:
                return f"Error switching to CRUISE mode: {result['error']}"

    data = client.navigate(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    nav = data.get("nav", {})
    fuel = data.get("fuel", {})
    wait_secs = _parse_arrival(nav)

    # Check if DRIFT was actually used
    flight_mode = nav.get("flightMode", "CRUISE")
    if flight_mode == "DRIFT":
        result = f"⚠️  {ship_symbol} navigating to {waypoint_symbol} using DRIFT mode (VERY SLOW!)."
    else:
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
def plan_route(ship_symbol: str, destination: str) -> str:
    """Check if a ship can reach a destination. Shows distance, current fuel, and whether the trip is feasible. Use this before navigate_ship to avoid stranding a ship."""
    import math

    # Get ship info
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"

    nav = ship.get("nav", {})
    fuel = ship.get("fuel", {})
    current_wp = nav.get("waypointSymbol", "")
    system = nav.get("systemSymbol", "")

    if not system:
        return f"Error: Could not determine system for {ship_symbol}"

    # Get all waypoints in system to find coordinates
    waypoints = client.list_waypoints(system)
    if isinstance(waypoints, dict) and "error" in waypoints:
        return f"Error: Could not fetch waypoints: {waypoints['error']}"

    origin = None
    dest = None
    for wp in waypoints:
        if wp.get("symbol") == current_wp:
            origin = wp
        if wp.get("symbol") == destination:
            dest = wp

    if not origin:
        return f"Error: Could not find origin waypoint {current_wp}"
    if not dest:
        return f"Error: Could not find destination {destination} in system {system}"

    # Calculate distance
    dx = dest.get("x", 0) - origin.get("x", 0)
    dy = dest.get("y", 0) - origin.get("y", 0)
    distance = math.sqrt(dx * dx + dy * dy)

    fuel_current = fuel.get("current", 0)
    fuel_capacity = fuel.get("capacity", 0)

    # Rough fuel estimate (SpaceTraders: CRUISE costs ~distance fuel)
    est_fuel = max(1, round(distance))

    lines = [f"Route: {current_wp} → {destination}"]
    lines.append(f"  Distance: {distance:.1f} units")
    lines.append(f"  Fuel: {fuel_current}/{fuel_capacity}")
    lines.append(f"  Est. fuel cost (CRUISE): ~{est_fuel}")

    if fuel_capacity == 0:
        lines.append(f"  Solar powered — no fuel needed")
    elif fuel_current >= est_fuel:
        lines.append(f"  Feasible: YES ({fuel_current - est_fuel} fuel remaining after trip)")
    else:
        lines.append(f"  Feasible: NO (need ~{est_fuel - fuel_current} more fuel). Refuel first or use DRIFT mode (free but slow).")

    # Show destination traits
    traits = [t.get("symbol", "") for t in dest.get("traits", [])]
    if traits:
        lines.append(f"  Destination traits: {', '.join(traits)}")

    return "\n".join(lines)


@tool
def refuel_ship(ship_symbol: str) -> str:
    """Refuel a ship at the current waypoint. Auto-docks if in orbit. The waypoint must have a marketplace that sells fuel."""
    err = _ensure_dock(ship_symbol)
    if err:
        return f"Error: {err}"
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
    """Extract ores and minerals from an asteroid. Auto-orbits if docked. Ship must be at an asteroid with a mining laser. Cooldown applies between extractions."""
    # Check cooldown first
    cooldown = client.get_cooldown(ship_symbol)
    if isinstance(cooldown, dict) and not cooldown.get("error"):
        remaining = cooldown.get("remainingSeconds", 0)
        if remaining > 0:
            return f"Error: {ship_symbol} is on cooldown. {remaining}s remaining before next extraction."
    err = _ensure_orbit(ship_symbol)
    if err:
        return f"Error: {err}"
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
    """Sell cargo from a ship at the current waypoint's marketplace. Auto-docks if in orbit. Specify the trade symbol (e.g. IRON_ORE) and number of units to sell."""
    err = _ensure_dock(ship_symbol)
    if err:
        return f"Error: {err}"
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
    """Jettison (discard) cargo from a ship. Use this to make room when cargo is full and you don't want to sell. Only jettison non-contract stuff."""
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
    """Transfer cargo between two ships at the same waypoint. Auto-orbits both ships if needed. Use this to offload cargo from a miner to a larger cargo ship."""
    err = _ensure_orbit(from_ship)
    if err:
        return f"Error: {err}"
    err = _ensure_orbit(to_ship)
    if err:
        return f"Error: {err}"
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
def view_jump_gate(waypoint_symbol: str) -> str:
    """View jump gate connections from a waypoint. Shows which systems are reachable.

    Args:
        waypoint_symbol: Waypoint with jump gate (e.g., 'X1-KD26-J1')
    """
    # Extract system from waypoint (e.g., 'X1-KD26-J1' -> 'X1-KD26')
    system_symbol = '-'.join(waypoint_symbol.split('-')[:2])

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
    """Deliver goods to fulfill a contract. Auto-docks if in orbit. Ship must be at the contract's delivery waypoint with the required goods."""
    err = _ensure_dock(ship_symbol)
    if err:
        return f"Error: {err}"
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
#  Planning tool
# ──────────────────────────────────────────────

@tool
def update_plan(plan: str) -> str:
    """Write or update your current plan. The plan is shown to you at the start of every turn in [Current Plan]. Use this to record what you intend to do and why, track multi-step goals, and note important discoveries. The plan is also visible to the human operator who can edit it."""
    from pathlib import Path
    import time

    plan_file = Path("plan.txt")

    # Check if plan was just updated (within last 60 seconds)
    if plan_file.exists():
        mtime = plan_file.stat().st_mtime
        age_seconds = time.time() - mtime
        if age_seconds < 60:
            return (f"ERROR: Plan was updated {int(age_seconds)}s ago! "
                    f"STOP PLANNING and START EXECUTING the existing plan. "
                    f"Read [Current Plan] and take the next action.")

    plan_file.write_text(plan, encoding="utf-8")
    return f"Plan updated ({len(plan)} chars). NOW EXECUTE IT - do not plan again!"


# ──────────────────────────────────────────────
#  Tool registry
# ──────────────────────────────────────────────

# Tier 1: Essential tools for mining/selling/contracts gameplay
# Tools auto-handle dock/orbit, so those are excluded
TIER_1_TOOLS = [
    # Navigation & planning
    navigate_ship, refuel_ship, plan_route,
    # Mining
    extract_ore,
    # Trading & cargo
    sell_cargo, transfer_cargo, jettison_cargo,
    # Contracts
    accept_contract, deliver_contract, fulfill_contract, negotiate_contract,
    # Info (view_market includes shipyard data)
    scan_system, view_market, view_ships, find_waypoints, view_contracts,
    # Planning
    update_plan,
]

# All tools (tier 2) — includes advanced/exploration tools
ALL_TOOLS = [
    # Observation
    view_agent, view_contracts, view_ships, view_ship_details, view_cargo,
    scan_system, find_waypoints, view_market, view_jump_gate,
    # Ship operations (dock/orbit still available for edge cases)
    orbit_ship, dock_ship, navigate_ship, refuel_ship, plan_route,
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
    # Planning
    update_plan,
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
    """Get a tool function by its name. Searches ALL_TOOLS (superset)."""
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
