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
    """List all ships with location, fuel, status, cooldowns, and cargo. Use this to track your fleet."""
    data = client.list_ships()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    
    lines = []
    for s in data:
        symbol = s['symbol']
        nav = s.get("nav", {})
        fuel = s.get("fuel", {})
        cargo = s.get("cargo", {})
        
        # Check Cooldown
        cooldown = client.get_cooldown(symbol)
        cd_text = ""
        if isinstance(cooldown, dict) and not cooldown.get("error"):
            rem = cooldown.get("remainingSeconds", 0)
            if rem > 0:
                cd_text = f" [COOLDOWN: {rem}s]"

        # Check Transit
        status = nav.get('status', '?')
        arrival_text = ""
        if status == "IN_TRANSIT":
            arrival_seconds = _parse_arrival(nav)
            if arrival_seconds > 0:
                arrival_text = f" (Arriving in {int(arrival_seconds)}s)"

        lines.append(f"   {symbol} ({s.get('registration', {}).get('role', '?')})")
        lines.append(f"   Loc: {nav.get('waypointSymbol', '?')} ({status}){arrival_text}")
        lines.append(f"   Fuel: {fuel.get('current', 0)}/{fuel.get('capacity', 0)} | Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', 0)}{cd_text}")
        lines.append("")
        
    return "\n".join(lines)

def _buys_or_sells(m_data: dict, target: str):
    # Define logic for buying and selling
    buys = target in (m_data.get('imports', []) + m_data.get('exchange', []))
    sells = target in (m_data.get('exports', []) + m_data.get('exchange', []))
    hit = False
    match_reason = ""

    if buys or sells:
        hit = True
        actions = []
        if buys: actions.append("buys")
        if sells: actions.append("sells")
        
        # Joins with "and/or" if both are true, otherwise just the single action
        action_str = " and ".join(actions)
        match_reason = f"market {action_str} {target}"

    return hit, match_reason


@tool
def find_nearest(ship_symbol: str, target: str) -> str:
    """
    Find the nearest location for a specific need.
    Target can be:
    - A Trade Good (e.g. "FUEL", "IRON_ORE") -> Finds markets selling/buying it.
    - A Trait (e.g. "SHIPYARD", "MARKETPLACE") -> Finds waypoints with this trait.
    - A Type (e.g. "ASTEROID", "ENGINEERED_ASTEROID") -> Finds waypoints of this type.
    """
    import math
    
    # Get Ship Location
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"
    
    system_symbol = ship['nav']['systemSymbol']
    ship_x = ship['nav']['route']['destination']['x']
    ship_y = ship['nav']['route']['destination']['y']

    # 1. Fetch all waypoints in system (this is cached by the client usually, or cheap)
    all_waypoints = client.list_waypoints(system_symbol)
    if isinstance(all_waypoints, dict) and "error" in all_waypoints:
        return f"Error: {all_waypoints['error']}"

    candidates = []
    target = target.upper()
    
    # 2. Search Logic
    market_cache = load_market_cache() # Use your existing cache loader
    
    for wp in all_waypoints:
        wp_sym = wp['symbol']
        dist = math.sqrt((wp['x'] - ship_x)**2 + (wp['y'] - ship_y)**2)
        
        hit = False
        match_reason = ""

        # Check Type
        if wp['type'] == target:
            hit = True
            match_reason = f"Type: {target}"
            
        # Check Traits
        traits = [t['symbol'] for t in wp.get('traits', [])]
        if target in traits:
            hit = True
            match_reason = f"Trait: {target}"

        if not hit and "MARKETPLACE" in traits:
            m_data = market_cache.get(wp_sym, {})
            hit, match_reason = _buys_or_sells(m_data, target)

        if hit:
            candidates.append((dist, wp_sym, match_reason))

    # 3. Sort and Return
    if not candidates:
        return f"No locations found for '{target}' in {system_symbol}."

    candidates.sort(key=lambda x: x[0]) # Sort by distance
    
    lines = [f"Found {len(candidates)} locations for '{target}' near {ship_symbol}:"]
    for dist, sym, reason in candidates[:5]:
        lines.append(f"- {sym}: {dist:.1f} distance ({reason})")
        
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


def _calculate_travel_cost(ship: dict, dest_wp: dict, origin_wp: dict, mode: str = "CRUISE") -> tuple[int, int, int]:
    """
    Helper to calculate distance, fuel cost, and estimated time.
    Returns: (distance, fuel_cost, flight_seconds)
    """
    import math
    
    # 1. Calculate Distance
    dx = dest_wp['x'] - origin_wp['x']
    dy = dest_wp['y'] - origin_wp['y']
    distance = math.sqrt(dx**2 + dy**2)
    
    # 2. Get Ship Engine Speed (Default to 30 for Command Ships if missing)
    # Satellites usually have speed 10, Command ships 30, Interceptors >30
    engine_speed = ship.get('engine', {}).get('speed', 30)
    
    # 3. Determine Multipliers
    # CRUISE: 1x fuel, 1x speed
    # DRIFT:  1 fuel total, 0.1x speed (relative to current engine?) - actually Drift is usually fixed low speed
    # BURN:   2x fuel, 2x speed 
    # STEALTH: 1x fuel, 1x speed 
    
    fuel_multiplier = 1.0
    speed_multiplier = 1.0
    
    # Round distance for fuel calc
    base_fuel_cost = max(1, round(distance))

    if mode == "DRIFT":
        fuel_cost = 1 
        # Drift is universally slow, often ignoring engine speed, but let's assume a penalty
        # In SpaceTraders, Drift is usually ~1/10th speed or fixed low speed.
        speed_multiplier = 0.01 # Severe penalty
    elif mode == "BURN":
        fuel_cost = 2 * base_fuel_cost
        speed_multiplier = 2.0
    else: # CRUISE or STEALTH
        fuel_cost = base_fuel_cost
        speed_multiplier = 1.0

    # 4. Handle Solar/Probe Ships (0 Fuel Capacity)
    fuel_capacity = ship.get('fuel', {}).get('capacity', 0)
    if fuel_capacity == 0:
        fuel_cost = 0  # Solar ships don't consume fuel units

    # 5. Calculate Time
    # SpaceTraders formula approximation:
    # time = round(max(1, distance) * (Multiplier / EngineSpeed)) + 15
    # Note: Multiplier in API docs is often "Distance / Speed" logic.
    # The accepted formula for CRUISE is roughly: (Distance * (1 / Speed)) + 15s (cooldown/warmup)
    
    # Fixed formula based on observation:
    # Travel Time = (Distance * (Multiplier / EngineSpeed) ) + 15
    # Where Multiplier for CRUISE is usually just 1 (implied).
    
    # However, for accurate estimates, we use the standard formula:
    # time = round(round(max(1, distance)) * (multiplier_const / engine_speed) + 15)
    # multiplier_const: CRUISE=1, DRIFT=100?, BURN=0.5?
    
    # Let's use the empirical observation logic:
    if mode == "BURN":
        flight_mode_mult = 0.5 # Faster
    elif mode == "DRIFT":
        flight_mode_mult = 5.0 # Slower (The API might differ, but this is a safer estimate)
    else:
        flight_mode_mult = 1.0 # Cruise/Stealth

    # Improve calculation:
    flight_time = round((max(1, distance) * (flight_mode_mult / max(1, engine_speed/100) )) + 15)
    
    # Simpler heuristic that matches your log (36 distance / speed 10 satellite ≈ 113s?)
    # 36 distance. 113s total. 15s is constant. 
    # Travel part = 98s. 
    # 98 / 36 = 2.72 seconds per unit.
    # Speed 10 = ~3s per unit. Speed 30 = ~1s per unit.
    # Formula: (Distance * (30 / Speed)) + 15
    
    travel_time = (distance * (30 / max(1, engine_speed)))
    
    if mode == "BURN":
        travel_time /= 2
    elif mode == "DRIFT":
        travel_time *= 5 # Drift is significantly slower
        
    total_time = travel_time + 15
    
    return round(distance), int(fuel_cost), int(total_time)

@tool
def navigate_ship(ship_symbol: str, waypoint_symbol: str, mode: str = "CRUISE") -> str:
    """Navigate a ship to a different waypoint. Consumes fuel and takes time.

    Args:
        ship_symbol: The ship to navigate
        waypoint_symbol: The destination waypoint (e.g., X1-AB12-C3)
        mode: Flight mode. 
              'CRUISE' (Standard, 1x fuel), 
              'DRIFT' (Slow, 1 fuel total), 
              'BURN' (Fast, 2x fuel), 
              'STEALTH' (Standard, reduced visibility)
    """
    import math

    # 1. Basic Ship Checks
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"

    mode = mode.upper()
    valid_modes = ["CRUISE", "DRIFT", "BURN", "STEALTH"]
    if mode not in valid_modes:
        return f"Error: Invalid mode {mode}. Must be one of {valid_modes}"

    nav = ship.get("nav", {})
    fuel = ship.get("fuel", {})
    current_wp_symbol = nav.get("waypointSymbol", "")
    system_symbol = nav.get("systemSymbol", "")
    fuel_current = fuel.get("current", 0)
    fuel_capacity = fuel.get("capacity", 0)
    
    if current_wp_symbol == waypoint_symbol:
         return f"{ship_symbol} is already at {waypoint_symbol}."

    # 2. Pre-Flight Calculation & Checks
    if fuel_capacity > 0: # Only check fuel for non-solar ships
        waypoints = client.list_waypoints(system_symbol)
        if isinstance(waypoints, dict) and "error" in waypoints:
             return f"Error fetching waypoints: {waypoints['error']}"

        origin = next((w for w in waypoints if w['symbol'] == current_wp_symbol), None)
        dest = next((w for w in waypoints if w['symbol'] == waypoint_symbol), None)

        if not origin or not dest:
            return f"Error: Cannot find coordinates for route {current_wp_symbol} -> {waypoint_symbol}"

        distance, fuel_cost, _ = _calculate_travel_cost(ship, dest, origin, mode)

        if fuel_current < fuel_cost:
            # Not enough fuel logic
            market_cache = load_market_cache()
            current_market = market_cache.get(current_wp_symbol, {})
            has_fuel = "FUEL" in current_market.get("exchange", []) or "FUEL" in current_market.get("exports", [])

            err_msg = (f"Error: Not enough fuel! {ship_symbol} needs {fuel_cost} for {mode} but has {fuel_current}.\n")
            
            if has_fuel:
                err_msg += f"You are at {current_wp_symbol} which sells FUEL. Call refuel_ship('{ship_symbol}') first!"
            elif mode != "DRIFT":
                # Check if DRIFT is an option (cost is always 1)
                err_msg += f"Try using mode='DRIFT' (costs 1 fuel) if you are stranded, or find a fuel station."
            
            return err_msg

    # 3. Execution
    err = _ensure_orbit(ship_symbol)
    if err: return f"Error: {err}"

    # Only change mode if necessary to save API calls
    if nav.get("flightMode") != mode:
        client.set_flight_mode(ship_symbol, mode)

    data = client.navigate(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    nav = data.get("nav", {})
    fuel = data.get("fuel", {})
    wait_secs = _parse_arrival(nav)

    # Store wait time
    navigate_ship._last_wait = wait_secs
    
    return (
        f"🚀 {ship_symbol} navigating to {waypoint_symbol} ({mode}).\n"
        f"Fuel: {fuel.get('current', 0)}/{fuel.get('capacity', 0)} (consumed ~{fuel.get('consumed', {}).get('amount', '?')})\n"
        f"Arrival in {int(wait_secs)}s."
    )

navigate_ship._last_wait = 0.0


@tool
def plan_route(ship_symbol: str, destination: str, mode: str = "CRUISE") -> str:
    """Calculate distance and fuel cost for a specific flight mode. Use this to check feasibility before flying."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"

    mode = mode.upper()
    nav = ship.get("nav", {})
    fuel = ship.get("fuel", {})
    current_wp_symbol = nav.get("waypointSymbol", "")
    system_symbol = nav.get("systemSymbol", "")
    
    if current_wp_symbol == destination:
        return f"Ship is already at {destination}."

    waypoints = client.list_waypoints(system_symbol)
    if isinstance(waypoints, dict) and "error" in waypoints:
         return f"Error fetching waypoints: {waypoints['error']}"
         
    origin = next((w for w in waypoints if w['symbol'] == current_wp_symbol), None)
    dest = next((w for w in waypoints if w['symbol'] == destination), None)

    if not origin or not dest:
        return f"Error finding waypoints."

    # Use shared helper
    dist, cost, time = _calculate_travel_cost(ship, dest, origin, mode)
    
    fuel_current = fuel.get("current", 0)
    fuel_cap = fuel.get("capacity", 0)
    
    lines = [f"Route: {current_wp_symbol} → {destination} ({mode})"]
    lines.append(f"  Distance: {dist} units")
    lines.append(f"  Est Time: {time}s")
    lines.append(f"  Fuel Cost: {cost}")
    lines.append(f"  Current Fuel: {fuel_current}/{fuel_cap}")

    if fuel_cap == 0:
        lines.append(f"  Fuel Cost: 0 (Solar Powered)")
        lines.append(f"  ✅ Feasible (Solar)")
    else:
        lines.append(f"  Fuel Cost: {cost}")
        if fuel_current >= cost:
            lines.append(f"  ✅ Feasible ({fuel_current - cost} remaining)")
        else:
            lines.append(f"  ❌ Insufficient Fuel (Short by {cost - fuel_current})")
            if mode != "DRIFT":
                lines.append(f"     Tip: Check plan_route with mode='DRIFT' (Cost: 1 fuel)")

    # Show traits of destination
    traits = [t['symbol'] for t in dest.get('traits', [])]
    if traits:
        lines.append(f"  Dest Traits: {', '.join(traits)}")

    return "\n".join(lines)


@tool
def refuel_ship(ship_symbol: str) -> str:
    """Refuel a ship at the current waypoint. The waypoint must have a marketplace that sells fuel."""
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


def _get_available_units(ship_symbol: str, trade_symbol: str, requested_units: int = None) -> tuple[int, str]:
    """
    Helper to determine safe unit counts for cargo operations.
    Returns: (safe_units, error_message)
    """
    # 1. Fetch live cargo data
    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        return 0, f"Error checking cargo for {ship_symbol}: {cargo_data['error']}"

    # 2. Find the specific item in inventory
    inventory = cargo_data.get("inventory", [])
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    # 3. Handle 0 availability immediately
    if available == 0:
        return 0, f"Error: Ship {ship_symbol} has no {trade_symbol} available."

    # 4. Determine final unit count (Magic Clamping)
    if requested_units is None:
        # Default to all
        final_units = available
    else:
        # Clamp requested amount to what is actually available
        final_units = min(requested_units, available)

    return final_units, None


def _get_contract_goods() -> set[str]:
    """Return the set of trade symbols required by any active, unfulfilled contract."""
    contracts = client.list_contracts()
    if not isinstance(contracts, list):
        return set()
    goods = set()
    for c in contracts:
        if c.get("accepted") and not c.get("fulfilled"):
            for d in c.get("terms", {}).get("deliver", []):
                goods.add(d["tradeSymbol"])
    return goods


@tool
def sell_cargo(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False) -> str:
    """Sell cargo. If units is None or exceeds inventory, sells all available."""
    # Guard: warn if this is a contract good
    # disable force
    if True:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            return (
                f"Error: {trade_symbol} is required by an active contract. "
                f"Use deliver_contract to deliver it instead. "
                f"Call sell_cargo with force=True to sell anyway."
            )

    safe_units, error = _get_available_units(ship_symbol, trade_symbol, units)
    if error:
        return error

    err = _ensure_dock(ship_symbol)
    if err:
        return f"Error: {err}"

    data = client.sell_cargo(ship_symbol, trade_symbol, safe_units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    tx = data.get("transaction", {})
    cargo = data.get("cargo", {})
    return (
        f"Sold {tx.get('units', safe_units)} {trade_symbol} for {tx.get('totalPrice', '?')} credits.\n"
        f"Cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


@tool
def jettison_cargo(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False) -> str:
    """Jettison cargo. If units is None or exceeds inventory, jettisons all available.

    If the item is required by an active contract, returns an error to prevent accidental loss.
    Pass force=True to jettison anyway.
    """
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            return (
                f"Error: {trade_symbol} is required by an active contract. "
                f"Use deliver_contract to deliver it instead. "
                f"Call jettison_cargo with force=True to jettison anyway."
            )

    safe_units, error = _get_available_units(ship_symbol, trade_symbol, units)
    if error:
        return error

    data = client.jettison(ship_symbol, trade_symbol, safe_units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    cargo = data.get("cargo", {})
    return (
        f"Jettisoned {safe_units} {trade_symbol}.\n"
        f"Cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


@tool
def transfer_cargo(from_ship: str, to_ship: str, trade_symbol: str, units: int = None) -> str:
    """Transfer cargo. Auto-orbits. If units is None or exceeds inventory, transfers all available."""
    
    # Check source ship inventory
    safe_units, error = _get_available_units(from_ship, trade_symbol, units)
    if error:
        return error

    # Ensure orbit states
    if err := _ensure_orbit(from_ship): return f"Error: {err}"
    if err := _ensure_orbit(to_ship): return f"Error: {err}"
    
    data = client.transfer_cargo(from_ship, to_ship, trade_symbol, safe_units)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
        
    cargo = data.get("cargo", {})
    return (
        f"Transferred {safe_units} {trade_symbol} from {from_ship} to {to_ship}.\n"
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
    """Deliver goods to fulfill a contract. Ship must be at the contract's delivery waypoint with the required goods."""
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
#  Behavior assignment tools
# ──────────────────────────────────────────────


@tool
def assign_mining_loop(ship_symbol: str, asteroid_wp: str, ore_types: str = "") -> str:
    """Assign an automated MINING_LOOP behavior to a ship.

    The ship will orbit the asteroid, continuously extract ore, and jettison
    non-target ores. You will only be called when cargo is full or an error occurs.
    Do NOT manually navigate or mine a ship that has an active MINING_LOOP.

    Args:
        ship_symbol: Ship to assign (must already be at the asteroid waypoint).
        asteroid_wp: Waypoint symbol of the asteroid to mine.
        ore_types: Comma-separated ore symbols to KEEP (e.g. "IRON_ORE,COPPER_ORE").
                   Everything else is jettisoned automatically.
                   Leave empty to keep all ores (you will be alerted when cargo is full).
    """
    from behaviors import get_engine
    params: dict = {"asteroid_wp": asteroid_wp}
    if ore_types:
        params["ore_types"] = [s.strip() for s in ore_types.split(",") if s.strip()]
    get_engine().assign(ship_symbol, "MINING_LOOP", params)
    ore_desc = f"keeping {params['ore_types']}" if ore_types else "keeping all ores"
    return f"Assigned MINING_LOOP to {ship_symbol}: mining at {asteroid_wp}, {ore_desc}."


@tool
def cancel_behavior(ship_symbol: str) -> str:
    """Cancel a ship's automated behavior and return it to manual LLM control.

    Use this when a ship needs manual direction (e.g. navigate to sell cargo,
    or when a behavior raised an alert that requires repositioning).
    """
    from behaviors import get_engine
    engine = get_engine()
    if ship_symbol not in engine.behaviors:
        return f"{ship_symbol} has no assigned behavior."
    engine.cancel(ship_symbol)
    return f"Cancelled behavior for {ship_symbol}. Ship is now idle (manual control)."


@tool
def assign_satellite_scout(ship_symbols: str, market_waypoints: str = "") -> str:
    """Assign SATELLITE_SCOUT behavior to one or more satellites.

    Satellites cycle through a market list, keeping price data fresh. They are
    distributed evenly so each starts at a different offset with minimal overlap.

    Args:
        ship_symbols: Comma-separated satellite ship symbols (e.g. "WHATER-2,WHATER-3").
        market_waypoints: Optional comma-separated waypoints to scout
                          (e.g. "X1-AB12-A1,X1-AB12-B2"). If omitted, the satellite
                          will automatically use all known markets from the cache,
                          picking up newly discovered markets over time.
    """
    from behaviors import get_engine
    ships = [s.strip() for s in ship_symbols.split(",") if s.strip()]
    markets = [m.strip() for m in market_waypoints.split(",") if m.strip()]

    if not ships:
        return "Error: no ship symbols provided."

    engine = get_engine()
    n = len(ships)

    if markets:
        m = len(markets)
        for i, ship in enumerate(ships):
            start_index = (i * m) // n
            engine.assign(ship, "SATELLITE_SCOUT", {
                "market_list": markets,
                "current_index": start_index,
            })
        offsets = [str((i * m) // n) for i in range(n)]
        return (
            f"Assigned SATELLITE_SCOUT to {n} satellite(s) across {m} markets. "
            f"Starting indices: {', '.join(f'{s}@{o}' for s, o in zip(ships, offsets))}."
        )
    else:
        # No explicit list — each satellite will use the full market cache dynamically.
        # Distribute starting offsets based on current cache size as a hint.
        cache = load_market_cache()
        cache_size = max(len(cache), 1)
        for i, ship in enumerate(ships):
            start_index = (i * cache_size) // n
            engine.assign(ship, "SATELLITE_SCOUT", {
                "market_list": [],  # empty = use full cache each tick
                "current_index": start_index,
            })
        return (
            f"Assigned SATELLITE_SCOUT to {n} satellite(s) using all known markets "
            f"({cache_size} currently in cache, list updates automatically)."
        )


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
    # Behavior control
    assign_mining_loop, assign_satellite_scout, cancel_behavior,
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
    # Find stuff
    find_nearest,
    # Behavior control
    assign_mining_loop, assign_satellite_scout, cancel_behavior,
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
