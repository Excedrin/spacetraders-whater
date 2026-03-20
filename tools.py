import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ship_status import FleetTracker

from dotenv import load_dotenv
from langchain_core.tools import tool

from api_client import SpaceTradersClient

# ──────────────────────────────────────────────
#  Global Config & Init
# ──────────────────────────────────────────────
load_dotenv()
client = SpaceTradersClient(os.environ["ST_TOKEN"])

TRADE_PROFIT_MARGIN = 0.10  # 10% margin for planned trades
JIT_TRADE_MARGIN = 0.02     # 2% safety margin for JIT trades
GATE_MIN_CREDIT_BUFFER = 250_000 
RESERVE_BUFFER = 350_000

# Fleet tracker — injected by the caller (bot.py or play_cli.py) via set_fleet().
_fleet_impl: "FleetTracker | None" = None


def get_fleet() -> "FleetTracker":
    """Get the fleet tracker. Raises error if not initialized.

    Returns:
        The FleetTracker instance

    Raises:
        RuntimeError: If set_fleet() was not called first
    """
    if _fleet_impl is None:
        raise RuntimeError(
            "Fleet tracker not initialized. Call set_fleet() before using tools."
        )
    return _fleet_impl


def set_fleet(f: "FleetTracker"):
    """Initialize the fleet tracker instance.

    Args:
        f: FleetTracker instance to use for ship state management
    """
    global _fleet_impl
    _fleet_impl = f
    # Also initialize strategy module with fleet reference
    try:
        from strategy import (
            set_fleet as set_strategy_fleet,
            set_hq_managed_ships as set_strategy_hq
        )
        set_strategy_fleet(f)
        # Sync HQ settings if already initialized
        global _hq_managed_ships
        set_strategy_hq(_hq_managed_ships)
    except ImportError:
        pass  # strategy.py not yet available


def try_get_fleet() -> "FleetTracker | None":
    """Try to get the fleet tracker, returning None if not initialized.

    Use this for optional operations where it's OK if the fleet isn't available.
    For required operations, use get_fleet() instead.

    Returns:
        The FleetTracker instance, or None if not initialized
    """
    return _fleet_impl


# Alert queue — injected by the server via set_alert_queue()
_alert_queue = []


def set_alert_queue(q):
    global _alert_queue
    _alert_queue = q


# HQ Director Toggle
# Can be "NONE", "ALL", or comma-separated list of ship roles/names
# Ship roles: HAULER, FREIGHTER, SATELLITE, COMMAND
# Ship names: WHATER-1, WHATER-2, etc.

_hq_managed_ships = os.environ.get("ST_ENABLE_HQ", "NONE").strip().upper()


def set_hq_enabled(targets: str):
    """Set which ships HQ should manage.

    Args:
        targets: "ALL" (all ships), "NONE" (disable), or comma-separated list of ship roles/names
                 Examples: "ALL", "NONE", "HAULER,SATELLITE", "WHATER-1,WHATER-2"
    """
    global _hq_managed_ships
    _hq_managed_ships = targets.strip().upper()


def get_hq_enabled() -> bool:
    """Returns True if HQ is enabled for any ships."""
    return _hq_managed_ships != "NONE"


def is_ship_hq_managed(ship_symbol: str, fleet) -> bool:
    """Check if a specific ship should be managed by HQ.

    Args:
        ship_symbol: The ship's symbol (e.g., "WHATER-1")
        fleet: FleetTracker instance to get ship role

    Returns:
        True if the ship should be HQ-managed
    """
    if _hq_managed_ships == "NONE":
        return False
    if _hq_managed_ships == "ALL":
        return True

    # Check if ship name or role is in the managed list
    ship = fleet.get_ship(ship_symbol) if fleet else None
    ship_role = ship.role if ship else None

    targets = {t.strip() for t in _hq_managed_ships.split(",")}
    return ship_symbol in targets or ship_role in targets


# ──────────────────────────────────────────────
#  Utility Helpers
# ──────────────────────────────────────────────

def get_system_from_waypoint(waypoint_symbol: str) -> str:
    """Extract the system symbol from a waypoint symbol (e.g., 'X1-ABC-123' -> 'X1-ABC')."""
    return waypoint_symbol.rsplit("-", 1)[0]


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def waypoint_distance(wp1_sym: str, wp2_sym: str, cache: dict = None) -> float:
    """Calculate distance between two waypoints using cached coordinates."""
    if not cache:
        cache = load_waypoint_cache()
    wp1 = cache.get(wp1_sym, {})
    wp2 = cache.get(wp2_sym, {})
    if not isinstance(wp1, dict) or not isinstance(wp2, dict):
        return float('inf')
    x1, y1 = wp1.get("x"), wp1.get("y")
    x2, y2 = wp2.get("x"), wp2.get("y")
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return float('inf')
    return calculate_distance(x1, y1, x2, y2)


WAYPOINT_CACHE_FILE = Path("waypoint_cache.json")
MARKET_CACHE_FILE = WAYPOINT_CACHE_FILE  # Alias for existing code
BEHAVIORS_FILE = Path("behaviors.json")
log = logging.getLogger(__name__)


def load_waypoint_cache() -> dict:
    """Load unified waypoint cache, with automatic migration from old market_cache.json."""
    if WAYPOINT_CACHE_FILE.exists():
        try:
            return json.loads(WAYPOINT_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    old_file = Path("market_cache.json")
    if old_file.exists():
        try:
            data = json.loads(old_file.read_text(encoding="utf-8"))
            WAYPOINT_CACHE_FILE.write_text(json.dumps(data, indent=2))
            return data
        except (json.JSONDecodeError, OSError):
            pass

    return {"_systems_fetched": []}


# Alias for backwards compatibility with existing code
load_market_cache = load_waypoint_cache


def _save_cache(cache: dict):
    """Persist unified waypoint cache to disk."""
    WAYPOINT_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _fetch_and_cache_construction(wp_sym: str):
    """Fetch construction status from API and store it in the waypoint cache."""
    sys_sym = wp_sym.rsplit("-", 1)[0]
    const = client.get_construction(sys_sym, wp_sym)
    if not isinstance(const, dict) or "error" in const:
        return
    cache = load_waypoint_cache()
    entry = cache.setdefault(wp_sym, {})
    entry["construction"] = {
        "isComplete": const.get("isComplete", False),
        "materials": const.get("materials", []),
    }
    _save_cache(cache)


def _ingest_waypoints(waypoints: list[dict]):
    """Ingest raw API waypoint objects into the unified cache.

    Merges all raw API data and determines traits like has_market, has_shipyard, is_charted.
    Also fetches construction status for newly discovered JUMP_GATEs.
    """
    cache = load_waypoint_cache()
    modified = False
    new_jump_gates = []
    for wp in waypoints:
        sym = wp.get("symbol")
        if not sym:
            continue
        entry = cache.setdefault(sym, {})
        is_new_gate = wp.get("type") == "JUMP_GATE" and "construction" not in entry
        entry.update(wp)  # Merge all raw API data

        # Determine traits and flags
        traits = [t.get("symbol", t) if isinstance(t, dict) else t for t in wp.get("traits", [])]
        entry["is_charted"] = "UNCHARTED" not in traits or wp.get("chart") is not None
        entry["has_market"] = "MARKETPLACE" in traits
        entry["has_shipyard"] = "SHIPYARD" in traits
        modified = True

        if is_new_gate:
            new_jump_gates.append(sym)

    if modified:
        _save_cache(cache)

    # Fetch construction status for newly discovered jump gates (after saving main cache)
    for jg_sym in new_jump_gates:
        _fetch_and_cache_construction(jg_sym)


def get_system_waypoints(system_symbol: str, waypoint_type: str = None, trait: str | None = None) -> list[dict]:
    """Get waypoints for a system, using cache if fully fetched, else API.

    On first call for a system, fetches all waypoints from API and caches them.
    Subsequent calls return cached data without API calls.

    Args:
        system_symbol: The system to fetch waypoints for
        waypoint_type: Optional filter by type (e.g., "JUMP_GATE", "ASTEROID")
        trait: Optional filter by trait (e.g., "MARKETPLACE", "SHIPYARD")
    """
    cache = load_waypoint_cache()
    fetched_systems = cache.setdefault("_systems_fetched", [])

    if system_symbol not in fetched_systems:
        wps = client.list_waypoints(system_symbol)
        if isinstance(wps, list) and wps:
            _ingest_waypoints(wps)
            cache = load_waypoint_cache()  # Reload with new data
            cache.setdefault("_systems_fetched", []).append(system_symbol)
            _save_cache(cache)
        else:
            return []  # Handle error gracefully

    # Return list from cache (exclude metadata)
    result = [v for k, v in cache.items() if k != "_systems_fetched" and k.startswith(system_symbol + "-")]

    # Apply filters if requested
    if waypoint_type:
        result = [w for w in result if w.get("type") == waypoint_type]
    if trait:
        result = [w for w in result if trait in [t.get("symbol", t) if isinstance(t, dict) else t for t in w.get("traits", [])]]

    return result


def _save_market_to_cache(waypoint_symbol: str, data: dict):
    """Save market data to cache. Uses unified _save_cache() helper."""
    import time

    cache = load_waypoint_cache()
    entry = cache.setdefault(waypoint_symbol, {})

    # Structural data (imports/exports/exchange)
    # Handle API variations where items might be dicts or strings
    for section in ("exports", "imports", "exchange"):
        items = data.get(section, [])
        if items:
            entry[section] = [
                i.get("symbol") if isinstance(i, dict) else str(i) for i in items
            ]

    # Explicitly check for MARKETPLACE trait to ensure existence is cached
    # This ensures discover_all_markets populates the cache keys even if goods data is missing
    traits = data.get("traits", [])
    trait_symbols = [t.get("symbol") if isinstance(t, dict) else str(t) for t in traits]
    if "MARKETPLACE" in trait_symbols:
        entry["is_market"] = True  # Marker flag
        entry["has_market"] = True  # Unified flag

    # Price data
    trade_goods = data.get("tradeGoods", [])
    if trade_goods:
        entry["trade_goods"] = [
            {
                "symbol": g["symbol"],
                "purchasePrice": g.get("purchasePrice"),
                "sellPrice": g.get("sellPrice"),
                "tradeVolume": g.get("tradeVolume"),
            }
            for g in trade_goods
        ]
        entry["last_updated"] = int(time.time())

    # Save if we have any relevant data
    if entry:
        cache[waypoint_symbol] = entry
        _save_cache(cache)


def _save_shipyard_to_cache(waypoint_symbol: str, data: dict):
    """Save shipyard pricing data to cache. Uses unified _save_cache() helper."""
    cache = load_waypoint_cache()
    entry = cache.setdefault(waypoint_symbol, {})

    ships = data.get("ships", data.get("shipTypes", []))
    if ships:
        # Filter down to essential fields to save space
        clean_ships = []
        for s in ships:
            if isinstance(s, dict):
                clean_ships.append(
                    {
                        "type": s.get("type"),
                        "name": s.get("name"),
                        "purchasePrice": s.get("purchasePrice"),
                    }
                )
        entry["ships"] = clean_ships
        entry["has_shipyard"] = True

    if entry:
        cache[waypoint_symbol] = entry
        _save_cache(cache)


# ──────────────────────────────────────────────
#  Payload Interceptors (State Synchronization)
# ──────────────────────────────────────────────


def _get_local_ship(ship_symbol: str):
    """Get ship from local tracker, eliminating GET /my/ships requests.

    Checks FleetTracker first for instant state. Only falls back to API
    if ship is not yet tracked.
    """
    if (fleet := try_get_fleet()):
        ship = fleet.get_ship(ship_symbol)
        if ship:
            return ship

    # Absolute fallback if ship is not in tracker
    raw = client.get_ship(ship_symbol)
    if isinstance(raw, dict) and "error" not in raw:
        if (fleet := try_get_fleet()):
            fleet.update_from_api([raw])
            return fleet.get_ship(ship_symbol)
    raise Exception(f"Ship {ship_symbol} not found.")


def _intercept_agent(data: dict):
    """Intercepts action payloads to update local agent state (e.g., credits)."""
    if isinstance(data, dict) and "agent" in data:
        get_fleet().update_agent(data["agent"])


def _get_local_agent() -> dict:
    """Get agent from local tracker, eliminating GET /my/agent requests."""
    if get_fleet().agent:
        return get_fleet().agent

    # Absolute fallback if agent is not in tracker
    raw = client.get_agent()
    if isinstance(raw, dict) and "error" not in raw:
        if (fleet := try_get_fleet()):
            fleet.update_agent(raw)
        return raw
    return {}


def _intercept(ship_symbol: str, data: dict):
    """Pipes action payloads directly into the local fleet tracker.

    Called after any action (dock, orbit, navigate, buy, sell, extract, etc.)
    to update local state without polling the API.
    """
    if isinstance(data, dict) and "error" not in data and (fleet := try_get_fleet()):
        fleet.update_ship_partial(ship_symbol, data)
        _intercept_agent(data)


# ──────────────────────────────────────────────
#  Core Logic Helpers (Shared)
# ──────────────────────────────────────────────
# These are pure logic functions used by both Tools and Behaviors.
# They return structured data (Exceptions or Tuples) rather than user-facing strings.


def _ensure_orbit_logic(ship_symbol: str) -> None:
    """Raises Exception if fails."""
    ship = _get_local_ship(ship_symbol)

    if ship.nav_status == "IN_ORBIT":
        return
    if ship.nav_status == "IN_TRANSIT":
        raise Exception(f"{ship_symbol} is currently in transit")

    result = client.orbit(ship_symbol)
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"Could not orbit {ship_symbol}: {result['error']}")

    _intercept(ship_symbol, result)


class PriceFloorHit(Exception):
    """Raised when market price drops below minimum acceptable price."""

    def __init__(self, trade_symbol, current_price, min_price, sold, revenue):
        self.trade_symbol = trade_symbol
        self.current_price = current_price
        self.min_price = min_price
        self.sold = sold
        self.revenue = revenue
        super().__init__(
            f"{trade_symbol} price {current_price} dropped below floor {min_price}. Sold {sold} for {revenue} cr before stopping."
        )


class PriceCeilingHit(Exception):
    """Raised when market price exceeds maximum acceptable buy price."""

    def __init__(self, trade_symbol, current_price, max_price, bought, spent):
        self.trade_symbol = trade_symbol
        self.current_price = current_price
        self.max_price = max_price
        self.bought = bought
        self.spent = spent
        super().__init__(
            f"{trade_symbol} price {current_price} exceeded ceiling {max_price}. Bought {bought} for {spent} cr before stopping."
        )


class MinQtyNotMet(Exception):
    """Raised when buy step could not acquire the minimum required quantity."""

    def __init__(self, trade_symbol, bought, min_qty, reason=""):
        self.trade_symbol = trade_symbol
        self.bought = bought
        self.min_qty = min_qty
        self.reason = reason
        super().__init__(
            f"{trade_symbol}: only bought {bought} (need at least {min_qty}). {reason}".strip()
        )


def _ensure_dock_logic(ship_symbol: str) -> None:
    """Raises Exception if fails."""
    ship = _get_local_ship(ship_symbol)

    if ship.nav_status == "DOCKED":
        return
    if ship.nav_status == "IN_TRANSIT":
        raise Exception(f"{ship_symbol} is currently in transit")

    result = client.dock(ship_symbol)
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"Could not dock {ship_symbol}: {result['error']}")

    _intercept(ship_symbol, result)


def _parse_arrival(nav: dict) -> float:
    route = nav.get("route", {})
    arrival_str = route.get("arrival")
    if not arrival_str or nav.get("status") != "IN_TRANSIT":
        return 0.0
    arrival = datetime.fromisoformat(arrival_str.replace("Z", "+00:00"))
    remaining = (arrival - datetime.now(timezone.utc)).total_seconds()
    return max(remaining, 0.0)


def _get_contract_goods() -> set[str]:
    contracts = client.list_contracts()
    if not isinstance(contracts, list):
        return set()
    goods = set()
    for c in contracts:
        if c.get("accepted") and not c.get("fulfilled"):
            for d in c.get("terms", {}).get("deliver", []):
                goods.add(d["tradeSymbol"])
    return goods


def evaluate_fleet_strategy(system_symbol: str | None = None) -> dict:
    """Core logic for game phase, budget, and fleet needs. Single source of truth."""
    agent = _get_local_agent()
    credits = agent.get("credits", 0)

    # Read instantly from local memory instead of API
    ships = list(get_fleet().ships.values()) if try_get_fleet() else []

    total_ships = len(ships)
    trader_count = sum(
        1 for s in ships
        if s.role in ["COMMAND", "HAULER", "FREIGHTER"]
    )

    probe_count = sum(
        1 for s in ships
        if s.role == "SATELLITE"
    )

    reserve_needed = max(trader_count * RESERVE_BUFFER, RESERVE_BUFFER)
    excess = credits - reserve_needed

    search_sys = system_symbol
    if not search_sys and ships and ships[0].location:
        search_sys = get_system_from_waypoint(ships[0].location)

    # --- Check Jump Gate Status ---
    hq_sys = get_system_from_waypoint(agent.get("headquarters", "")) if agent.get("headquarters") else search_sys
    gate_built = False
    needs_gate_materials = False
    cache = load_waypoint_cache()

    if hq_sys:
        for wp_sym, wp_data in cache.items():
            if not isinstance(wp_data, dict) or not wp_sym.startswith(hq_sys + "-"):
                continue
            if wp_data.get("type") == "JUMP_GATE":
                if "construction" not in wp_data:
                    _fetch_and_cache_construction(wp_sym)
                    cache = load_waypoint_cache()
                    wp_data = cache.get(wp_sym, {})
                const = wp_data.get("construction", {})
                gate_built = const.get("isComplete", False)
                if not gate_built:
                    needs_gate_materials = True
                break

    # --- Phase Logic ---
    if trader_count < 2:
        phase = 1
        phase_name = "PHASE 1: BOOTSTRAP (Goal: Accumulate credits for first Hauler)"
    elif trader_count < 3:
        phase = 2
        phase_name = "PHASE 2: BUILDUP (Goal: Buy second Hauler to maximize local trade)"
    elif not gate_built:
        phase = 3
        phase_name = "PHASE 3: GATE CONSTRUCTION (Goal: Slowly fund and supply jump gate materials)"
    else:
        phase = 4
        phase_name = "PHASE 4: EXPANSION (Goal: Chart new systems, build massive fleet)"

    # Shipyard Info — Search globally for the cheapest ship prices
    hauler_price = float("inf")
    probe_price = float("inf")
    cheapest_shipyard = "Unknown"
    cheapest_probe_shipyard = "Unknown"
    fallback_shipyard = None

    # Count markets globally for probe scaling
    market_count = 0
    for wp, data in cache.items():
        if not isinstance(data, dict) or wp == "_systems_fetched":
            continue

        if data.get("has_market"):
            market_count += 1

        if "ships" in data:
            for s in data["ships"]:
                if s["type"] in ["SHIP_LIGHT_HAULER", "SHIP_HEAVY_FREIGHTER", "SHIP_COMMAND_FRIGATE"]:
                    if s.get("purchasePrice", float("inf")) < hauler_price:
                        hauler_price = s["purchasePrice"]
                        cheapest_shipyard = wp
                if s["type"] == "SHIP_PROBE":
                    if s.get("purchasePrice", float("inf")) < probe_price:
                        probe_price = s["purchasePrice"]
                        cheapest_probe_shipyard = wp
        elif data.get("has_shipyard") and not fallback_shipyard:
            fallback_shipyard = wp

    # If no priced shipyard found, use any known shipyard with default price
    if cheapest_shipyard == "Unknown" and fallback_shipyard:
        cheapest_shipyard = fallback_shipyard
    if cheapest_probe_shipyard == "Unknown" and fallback_shipyard:
        cheapest_probe_shipyard = fallback_shipyard

    ship_at_shipyard = False
    if cheapest_shipyard != "Unknown":
        ship_at_shipyard = any(s.location == cheapest_shipyard for s in ships)

    # --- Allow fleet expansion post-gate ---
    # Max 3 traders before gate, max 15 traders after gate.
    max_traders = 15 if gate_built else 2
    can_buy_ship = excess > hauler_price and trader_count < max_traders

    # Probe scaling: ~1 probe per 15 markets, minimum 1
    desired_probes = max(1, market_count // 10)
    needs_probe = probe_count < desired_probes and excess > probe_price

    return {
        "phase": phase,
        "phase_name": phase_name,
        "credits": credits,
        "reserve_needed": reserve_needed,
        "excess": excess,
        "trader_count": trader_count,
        "probe_count": probe_count,
        "desired_probes": desired_probes,
        "hauler_price": hauler_price,
        "probe_price": probe_price,
        "cheapest_shipyard": cheapest_shipyard,
        "cheapest_probe_shipyard": cheapest_probe_shipyard,
        "ship_at_shipyard": ship_at_shipyard,
        "can_buy_ship": can_buy_ship,
        "needs_probe": needs_probe,
        "needs_gate_materials": needs_gate_materials,
    }


def get_financial_assessment(system_symbol: str | None = None) -> str:
    """Calculates fleet budget requirements and recommends expansion/construction using the DRY strategy engine."""
    strat = evaluate_fleet_strategy(system_symbol)

    lines = [
        f"=== FLEET STRATEGY ===",
        f"{strat['phase_name']}",
        f"Current Credits: {strat['credits']:,} cr",
        f"Recommended Reserve: {strat['reserve_needed']:,} cr ({strat['trader_count']} traders)",
        f"Excess Capital: {strat['excess']:,} cr",
        ""
    ]

    if strat['excess'] > 3_000_000:
        lines.append(f"🟣 MASSIVE WEALTH: You have >3M credits. Focus entirely on Jump Gate construction.")
    elif strat['can_buy_ship']:
        lines.append(
            f"🟢 EXPANSION READY: You have enough excess capital to buy a Hauler (~{strat['hauler_price']:,} cr at {strat['cheapest_shipyard']})."
        )
        if not strat['ship_at_shipyard']:
            if strat['cheapest_shipyard'] != "Unknown":
                lines.append(f"   ⚠️ ACTION REQUIRED: Send a ship to {strat['cheapest_shipyard']} to make the purchase.")
            else:
                lines.append(f"   ⚠️ ACTION REQUIRED: Find a shipyard to make the purchase (no priced shipyards known in current system).")
    elif strat['excess'] > 0 and strat['trader_count'] >= 3:
        lines.append(
            f"🔵 FLEET CAPPED: Local market is saturated ({strat['trader_count']} traders). Excess funds will auto-route to Jump Gate Construction."
        )
    elif strat['excess'] > 0:
        lines.append(
            f"🟡 ACCUMULATING: Keep trading. Next goal: Hauler (~{strat['hauler_price']:,} cr)."
        )
    else:
        lines.append(
            f"🔴 LOW CAPITAL: Do not buy ships or materials! Focus on autotrade to build reserve."
        )

    lines.append(f"\n[HQ Fleet Director: {_hq_managed_ships}]")

    return "\n".join(lines)


def _calculate_travel_cost(
    ship: dict, dest_wp: dict, origin_wp: dict, mode: str = "CRUISE"
) -> tuple[int, int, int]:
    """
    Helper to calculate distance, fuel cost, and estimated time.
    Returns: (distance, fuel_cost, flight_seconds)
    """
    # 1. Calculate Distance
    distance = calculate_distance(
        origin_wp["x"], origin_wp["y"], dest_wp["x"], dest_wp["y"]
    )

    # 2. Get Ship Engine Speed (Default to 30 for Command Ships if missing)
    # Satellites usually have speed 10, Command ships 30, Interceptors >30
    engine_speed = ship.get("engine", {}).get("speed", 30)

    # 3. Determine Multipliers
    # CRUISE: 1x fuel, 1x speed
    # DRIFT:  1 fuel total, 0.1x speed (relative to current engine?) - actually Drift is usually fixed low speed
    # BURN:   2x fuel, 2x speed
    # STEALTH: 1x fuel, 1x speed

    # Round distance for fuel calc
    base_fuel_cost = max(1, round(distance))

    if mode == "DRIFT":
        fuel_cost = 1
        # Drift is universally slow, often ignoring engine speed, but let's assume a penalty
        # In SpaceTraders, Drift is usually ~1/10th speed or fixed low speed.
        speed_multiplier = 0.01  # Severe penalty
    elif mode == "BURN":
        fuel_cost = 2 * base_fuel_cost
        speed_multiplier = 2.0
    else:  # CRUISE or STEALTH
        fuel_cost = base_fuel_cost
        speed_multiplier = 1.0

    # 4. Handle Solar/Probe Ships (0 Fuel Capacity)
    fuel_capacity = ship.get("fuel", {}).get("capacity", 0)
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
        flight_mode_mult = 0.5  # Faster
    elif mode == "DRIFT":
        flight_mode_mult = (
            5.0  # Slower (The API might differ, but this is a safer estimate)
        )
    else:
        flight_mode_mult = 1.0  # Cruise/Stealth

    # Simpler heuristic that matches your log (36 distance / speed 10 satellite ≈ 113s?)
    # 36 distance. 113s total. 15s is constant.
    # Travel part = 98s.
    # 98 / 36 = 2.72 seconds per unit.
    # Speed 10 = ~3s per unit. Speed 30 = ~1s per unit.
    # Formula: (Distance * (30 / Speed)) + 15

    travel_time = distance * (30 / max(1, engine_speed))

    if mode == "BURN":
        travel_time /= 2
    elif mode == "DRIFT":
        travel_time *= 5  # Drift is significantly slower

    total_time = travel_time + 15

    return round(distance), int(fuel_cost), int(total_time)


def _find_refuel_path(
    ship: dict, origin_wp: dict, target_wp: dict, waypoints: list, mode="CRUISE"
) -> list[str] | None:
    """
    Perform BFS to find a path from origin to target using Marketplaces as refuel stops.
    Returns a list of waypoint symbols: [Origin, Stop1, Stop2, Target].
    """
    fuel_capacity = ship.get("fuel", {}).get("capacity", 0)
    current_fuel = ship.get("fuel", {}).get("current", 0)

    # 1. If solar (0 capacity), path is always direct.
    if fuel_capacity == 0:
        return [origin_wp["symbol"], target_wp["symbol"]]

    # 2. Identify potential stops (Marketplaces)
    # Assumption: All marketplaces sell fuel.
    potential_stops = [
        w
        for w in waypoints
        if "MARKETPLACE" in [t["symbol"] for t in w.get("traits", [])]
    ]

    # 3. BFS State: (current_wp_obj, path_list_of_symbols, fuel_at_current_node)
    queue = deque([(origin_wp, [origin_wp["symbol"]], current_fuel)])
    visited = {origin_wp["symbol"]}

    while queue:
        curr_node, path, fuel_available = queue.popleft()

        # A. Can we reach the Final Target from here?
        _, cost_to_target, _ = _calculate_travel_cost(ship, target_wp, curr_node, mode)
        if cost_to_target <= fuel_available:
            return path + [target_wp["symbol"]]

        # B. If not, find reachable Marketplaces to hop to
        for stop in potential_stops:
            if stop["symbol"] in visited:
                continue

            _, cost_to_stop, _ = _calculate_travel_cost(ship, stop, curr_node, mode)

            if cost_to_stop <= fuel_available:
                visited.add(stop["symbol"])
                # We assume we refuel to FULL CAPACITY at the stop
                queue.append((stop, path + [stop["symbol"]], fuel_capacity))

    return None


# ──────────────────────────────────────────────
#  CORE ACTION LOGIC (The "Smart" Layer)
# ──────────────────────────────────────────────


def _plan_route_logic(
    current_wp: str,
    destination_symbol: str,
    ship_dict: dict,
    mode: str,
) -> Tuple[str, float]:
    """Planning mode: returns route forecast without executing actions."""
    current_sys = get_system_from_waypoint(current_wp) if current_wp else ""
    dest_sys = get_system_from_waypoint(destination_symbol)
    is_inter_system = current_sys != dest_sys

    if is_inter_system:
        local_jgs = get_system_waypoints(current_sys, waypoint_type="JUMP_GATE")
        if not local_jgs or (isinstance(local_jgs, dict) and "error" in local_jgs):
            return f"Error: No Jump Gate found in current system {current_sys}.", 0.0

        local_jg_sym = local_jgs[0]["symbol"]

        lines = [f"Inter-System Route Plan: {current_wp} -> {destination_symbol}"]

        local_wps = get_system_waypoints(current_sys)
        if isinstance(local_wps, dict) and "error" in local_wps:
            return f"Error: {local_wps['error']}", 0.0

        origin_obj = next((w for w in local_wps if w["symbol"] == current_wp), None)
        local_jg_obj = next((w for w in local_wps if w["symbol"] == local_jg_sym), None)

        t1, c1 = 0, 0
        if current_wp != local_jg_sym and origin_obj and local_jg_obj:
            _, c1, t1 = _calculate_travel_cost(ship_dict, local_jg_obj, origin_obj, mode)

        lines.append(f"1. Nav to local gate ({local_jg_sym}): {t1}s, {c1} fuel")

        jg_data = client.get_jump_gate(current_sys, local_jg_sym)
        connections = (
            jg_data.get("connections", [])
            if isinstance(jg_data, dict) and "error" not in jg_data
            else []
        )
        connected_systems = [get_system_from_waypoint(cw) for cw in connections]

        if dest_sys in connected_systems:
            lines.append(f"2. Jump directly to {dest_sys} gate: Antimatter / Cooldown")
        else:
            lines.append(f"2. Multi-system Jump Route to {dest_sys}: (Auto-routing via nearest connected system)")

        lines.append(f"3. Nav to final dest ({destination_symbol}): Calculated upon arrival in {dest_sys}")
        return "\n".join(lines), 0.0

    # Intra-system planning
    waypoints = get_system_waypoints(current_sys)
    if isinstance(waypoints, dict) and "error" in waypoints:
        return f"Error: {waypoints['error']}", 0.0

    origin_obj = next((w for w in waypoints if w["symbol"] == current_wp), None)
    target_obj = next((w for w in waypoints if w["symbol"] == destination_symbol), None)

    if not origin_obj or not target_obj:
        return "Error: Could not resolve origin or destination coordinates.", 0.0

    lines = [f"Route Plan: {current_wp} -> {destination_symbol}"]
    dist, direct_cost, direct_time = _calculate_travel_cost(
        ship_dict, target_obj, origin_obj, "CRUISE"
    )
    lines.append(f"Direct Distance: {dist}")
    lines.append("\nFlight Modes:")

    fuel_capacity = ship_dict["fuel"]["capacity"]
    fuel_current = ship_dict["fuel"]["current"]

    for m in ["CRUISE", "DRIFT", "BURN"]:
        path = _find_refuel_path(ship_dict, origin_obj, target_obj, waypoints, mode=m)
        _, cost, time = _calculate_travel_cost(ship_dict, target_obj, origin_obj, m)

        status = ""
        if fuel_capacity > 0:
            if cost <= fuel_current:
                status = "✅ Direct"
            elif path:
                stops = len(path) - 2
                status = f"✅ Multi-hop ({stops} stops: {'->'.join(path)})"
            else:
                status = "❌ Impossible (Max range exceeded)"
        else:
            status = "✅ (Solar)"

        lines.append(
            f"  {m.ljust(7)}: {str(time).rjust(4)}s | Fuel: {str(cost).rjust(4)} | {status}"
        )

    return "\n".join(lines), 0.0


def _navigate_ship_logic(
    ship_symbol: str,
    destination_symbol: str,
    mode: str = "CRUISE",
    execute: bool = True,
) -> Tuple[str, float]:
    """
    Returns (result_message, wait_seconds).
    Handles smart routing, auto-refueling, and inter-system logic.
    """
    # 1. Structural Validation (Fast Fail)
    # A valid waypoint symbol MUST be SECTOR-SYSTEM-POINT (at least 2 hyphens).
    # This catches "WHATER-1" (ship) or "X1-RV42" (system only).
    if destination_symbol.count("-") < 2:
        raise Exception(
            f"Invalid destination format '{destination_symbol}'. Expected Waypoint Symbol (SECTOR-SYSTEM-WAYPOINT)."
        )

    ship_status = _get_local_ship(ship_symbol)
    current_wp = ship_status.location or ""
    current_sys = get_system_from_waypoint(current_wp) if current_wp else ""

    ship_dict = {
        "fuel": {"current": ship_status.fuel_current, "capacity": ship_status.fuel_capacity},
        "engine": {"speed": ship_status.engine_speed},
    }

    if not execute:
        return _plan_route_logic(current_wp, destination_symbol, ship_dict, mode)

    if current_wp == destination_symbol:
        return f"{ship_symbol} is already at {destination_symbol}.", 0.0

    # Inter-System Check
    dest_sys = get_system_from_waypoint(destination_symbol)
    target_wp_symbol = destination_symbol
    is_inter_system = current_sys != dest_sys

    if is_inter_system:
        # Get Local Jump Gate
        local_jgs = get_system_waypoints(current_sys, waypoint_type="JUMP_GATE")
        if not local_jgs or (isinstance(local_jgs, dict) and "error" in local_jgs):
            raise Exception(f"No Jump Gate found in current system {current_sys}.")

        local_jg_sym = local_jgs[0]["symbol"]

        if current_wp != local_jg_sym:
            # Step 1: Navigate to the local jump gate first.
            target_wp_symbol = local_jg_sym
        else:
            # Step 2: At local jump gate. Verify connection and Jump!
            _ensure_orbit_logic(ship_symbol)

            jg_data = client.get_jump_gate(current_sys, local_jg_sym)
            if isinstance(jg_data, dict) and "error" in jg_data:
                raise Exception(f"Failed to read jump gate: {jg_data['error']}")

            connections = jg_data.get("connections", [])

            # Map connections to their system symbols (sys -> exact waypoint)
            connected_systems = {}
            for cw in connections:
                cs = get_system_from_waypoint(cw)
                connected_systems[cs] = cw

            if dest_sys in connected_systems:
                # Directly connected! Jump to the specific waypoint linked to the destination system
                next_jump_target = connected_systems[dest_sys]
            else:
                # Greedy Pathfinding: Find connected system closest to ultimate destination
                dest_sys_data = client.get_system(dest_sys)
                if "error" in dest_sys_data:
                    raise Exception(
                        f"Failed to fetch destination system {dest_sys}: {dest_sys_data['error']}"
                    )

                dx, dy = dest_sys_data.get("x", 0), dest_sys_data.get("y", 0)
                best_dist = float("inf")
                next_jump_target = None

                for cs, cw in connected_systems.items():
                    cs_data = client.get_system(cs)
                    if "error" in cs_data:
                        continue
                    cx, cy = cs_data.get("x", 0), cs_data.get("y", 0)
                    dist = calculate_distance(cx, cy, dx, dy)

                    if dist < best_dist:
                        best_dist = dist
                        next_jump_target = cw

                if not next_jump_target:
                    raise Exception(
                        "Dead end: No jump connections available to route to destination."
                    )

            # Execute jump
            jump_res = client.jump(ship_symbol, next_jump_target)
            _intercept(ship_symbol, jump_res)
            if isinstance(jump_res, dict) and "error" in jump_res:
                raise Exception(
                    f"Jump to {next_jump_target} failed: {jump_res['error']}"
                )

            cd = jump_res.get("cooldown", {}).get("remainingSeconds", 0)
            if cd > 0:
                get_fleet().set_transit(ship_symbol, float(cd))

            next_sys = get_system_from_waypoint(next_jump_target)
            return (
                f"🚀 JUMPED to {next_sys} (Routing towards {dest_sys}). Cooldown: {cd}s.",
                float(cd),
            )

    # Fetch Waypoints to validate target exists and calculate stats
    waypoints = get_system_waypoints(current_sys)
    if isinstance(waypoints, dict) and "error" in waypoints:
        raise Exception(waypoints["error"])

    origin_obj = next((w for w in waypoints if w["symbol"] == current_wp), None)
    target_obj = next((w for w in waypoints if w["symbol"] == target_wp_symbol), None)

    # Local Existence Check
    if not origin_obj:
        raise Exception(f"Current location {current_wp} not found in system listing.")
    if not target_obj:
        # If we are staying in-system, this means the destination is bogus
        if not is_inter_system:
            raise Exception(
                f"Destination {target_wp_symbol} does not exist in system {current_sys}."
            )
        # If we are inter-system, target_obj is the Jump Gate, which must exist (checked above)
        raise Exception("Could not resolve Jump Gate coordinates.")

    # 1. SMART REFUEL AT DEPARTURE
    # Only refuel if the current market sells fuel AND we are not full.
    fuel_current = ship_status.fuel_current
    fuel_capacity = ship_status.fuel_capacity
    if fuel_capacity > 0:
        market_cache = load_market_cache()
        curr_market = market_cache.get(current_wp, {})
        has_fuel = "FUEL" in curr_market.get(
            "exchange", []
        ) or "FUEL" in curr_market.get("exports", [])

        if has_fuel and fuel_current < fuel_capacity:
            try:
                _refuel_ship_logic(ship_symbol)
                # Re-read from fleet tracker (synced via _intercept inside _refuel_ship_logic)
                ship_status = _get_local_ship(ship_symbol)
                fuel_current = ship_status.fuel_current
                fuel_capacity = ship_status.fuel_capacity
                ship_dict["fuel"] = {"current": fuel_current, "capacity": fuel_capacity}
            except Exception:
                pass

    _, direct_cost, direct_time = _calculate_travel_cost(
        ship_dict, target_obj, origin_obj, mode
    )

    # 2. Route Check
    next_hop = target_wp_symbol
    is_multi_hop = False

    if fuel_capacity > 0 and direct_cost > fuel_current:
        path = _find_refuel_path(ship_dict, origin_obj, target_obj, waypoints, mode)
        if not path:
            raise Exception(
                f"Stranded. Cannot reach {target_wp_symbol} ({direct_cost} fuel needed) and no refueling path found."
            )

        if len(path) > 1:
            next_hop = path[1]
            is_multi_hop = True
            next_hop_obj = next(
                (w for w in waypoints if w["symbol"] == next_hop), None
            )
            _, _, direct_time = _calculate_travel_cost(
                ship_dict, next_hop_obj, origin_obj, mode
            )

    # Action
    _ensure_orbit_logic(ship_symbol)
    if ship_status.flight_mode != mode:
        client.set_flight_mode(ship_symbol, mode)
        _intercept(ship_symbol, {"nav": {"flightMode": mode}})

    data = client.navigate(ship_symbol, next_hop)
    _intercept(ship_symbol, data)
    if isinstance(data, dict) and "error" in data:
        raise Exception(f"Error navigating: {data['error']}")

    wait_secs = _parse_arrival(data.get("nav", {}))
    # Navigate was called — transit MUST be > 0. If _parse_arrival returned 0
    # (e.g. arrival already past due to latency, or missing nav data),
    # fall back to the calculated estimate so transit always registers.
    if wait_secs <= 0:
        wait_secs = max(float(direct_time), 1.0)
    if (fleet := try_get_fleet()):
        fleet.set_transit(ship_symbol, wait_secs)
    result = f"🚀 {ship_symbol} navigating to {next_hop} ({mode}). Est: {direct_time}s."

    if is_multi_hop:
        result += f"\nNote: Multi-hop route initiated. Stopping at {next_hop} to refuel."
    elif is_inter_system:
        result += f"\nNote: En route to Jump Gate. Will execute system jump upon arrival."

    return result, wait_secs


def _extract_ore_logic(ship_symbol: str) -> Tuple[str, float]:
    """Returns (log_string, cooldown_seconds)."""
    # Check cooldown from fleet tracker (avoids an extra API call)
    if (fleet := try_get_fleet()):
        ship_status = fleet.get_ship(ship_symbol)
        if (
            ship_status
            and not ship_status.is_available()
            and ship_status.busy_reason == "extraction_cooldown"
        ):
            remaining = ship_status.seconds_until_available()
            return f"Cooldown remaining: {remaining:.0f}s", remaining

    _ensure_orbit_logic(ship_symbol)
    data = client.extract(ship_symbol)
    _intercept(ship_symbol, data)
    print(data)

    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        err_msg = str(err)

        # Handle API error 4000 (Cooldown) specifically if API returns it as error
        # Some versions of the API/Client might structure this differently
        if "cooldown" in err_msg.lower() or (
            isinstance(err, dict) and err.get("code") == 4000
        ):
            # Fallback guess
            return "Hit cooldown", 70.0

        # Normalize error message to avoid "0" or "4000"
        if isinstance(err, (int, float)):
            err_msg = f"API Error Code {err}"
        elif isinstance(err, dict) and "message" in err:
            err_msg = err["message"]

        raise Exception(err_msg)

    extraction = data.get("extraction", {})
    cd = data.get("cooldown", {})
    cargo = data.get("cargo", {})
    yielded = extraction.get("yield", {})

    result = f"Extracted {yielded.get('units', '?')} {yielded.get('symbol', '?')}."
    result += f" Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', '?')}."
    cd_secs = float(cd.get("remainingSeconds", 0))
    if cd_secs > 0 and (fleet := try_get_fleet()):
        get_fleet().set_extraction_cooldown(ship_symbol, cd_secs)
    return result, cd_secs


def _sell_cargo_logic(
    ship_symbol: str,
    trade_symbol: str,
    units: int | None = None,
    force: bool = False,
    min_price: int | None = None,
) -> str:
    # 1. Check Contract Safety
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            raise Exception(
                f"{trade_symbol} is required by an active contract. Use force=True to override."
            )

        # Check Gate Materials Safety
        if trade_symbol in ["FAB_MATS", "ADVANCED_CIRCUITRY"]:
            strat = evaluate_fleet_strategy()
            if strat.get("needs_gate_materials"):
                raise Exception(
                    f"{trade_symbol} is needed for Jump Gate construction! Use force=True to override."
                )

    # 2. Get ship state from local tracker (0 API calls)
    ship = _get_local_ship(ship_symbol)
    waypoint = ship.location

    available = 0
    for item in ship.cargo_inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        raise Exception(f"Ship {ship_symbol} has no {trade_symbol}.")

    target_units = available if units is None else min(units, available)

    # 3. Look up max transaction volume from market cache
    max_per_transaction = target_units
    if waypoint:
        cache = load_waypoint_cache()
        if waypoint in cache:
            for good in cache[waypoint].get("trade_goods", []):
                if good.get("symbol") == trade_symbol:
                    vol = good.get("tradeVolume")
                    if vol and vol > 0:
                        max_per_transaction = vol
                    break

    # 4. Enforce minimum price from cargo costs if not explicitly set
    avg_cost = ship.cargo_costs.get(trade_symbol, 0.0)
    if min_price is None and avg_cost > 0:
        min_price = int(avg_cost) + 1  # Never sell below cost

    # 5. Sell in chunks if needed
    _ensure_dock_logic(ship_symbol)
    total_sold = 0
    total_revenue = 0
    sell_count = 0
    previous_price_per_unit = None

    while total_sold < target_units:
        units_to_sell = min(max_per_transaction, target_units - total_sold)

        data = client.sell_cargo(ship_symbol, trade_symbol, units_to_sell)
        _intercept(ship_symbol, data)
        if isinstance(data, dict) and "error" in data:
            if sell_count == 0:
                raise Exception(data["error"])
            break  # Partial success — stop and report what we sold

        tx = data.get("transaction", {})
        units_sold = tx.get("units", units_to_sell)
        price_per_unit = tx.get("pricePerUnit", 0)

        total_sold += units_sold
        total_revenue += tx.get("totalPrice", 0)
        sell_count += 1

        # Update cargo costs on sell
        ship.update_cargo_costs_on_sell(trade_symbol)

        # 1. Did the price drop below our floor on THIS transaction?
        if min_price and price_per_unit < min_price:
            break

        # 2. The 1.25x Curve Prediction!
        if min_price and previous_price_per_unit is not None:
            price_delta = previous_price_per_unit - price_per_unit
            predicted_next_delta = price_delta * 1.25
            predicted_next_price = price_per_unit - predicted_next_delta

            if predicted_next_price < min_price:
                break

        previous_price_per_unit = price_per_unit

    cargo = data.get("cargo", {})
    if sell_count > 1:
        return f"Sold {total_sold} {trade_symbol} for {total_revenue} cr ({sell_count} transactions). Cargo: {cargo.get('units')}/{cargo.get('capacity')}."
    else:
        return f"Sold {total_sold} {trade_symbol} for {total_revenue} cr. Cargo: {cargo.get('units')}/{cargo.get('capacity')}."


def _buy_cargo_logic(
    ship_symbol: str,
    trade_symbol: str,
    units: int | None = None,
    max_price: int | None = None,
    min_qty: int | None = None,
) -> str:
    """Buy cargo from the current market. Splits purchases across multiple transactions if needed.

    Respects cargo capacity, credits available, market transaction limits, and optional price/quantity guards.
    Uses local FleetTracker state to avoid GET /my/ships calls.
    """
    # Get ship state from local tracker (0 API calls)
    ship = _get_local_ship(ship_symbol)
    waypoint = ship.location
    if not waypoint:
        raise Exception(f"Cannot determine ship location")

    capacity = ship.cargo_capacity
    current_units = ship.cargo_units
    available_space = capacity - current_units

    if available_space <= 0:
        raise Exception(
            f"Ship {ship_symbol} cargo is full ({current_units}/{capacity}). No space for purchase."
        )

    # Determine target amount (cargo-limited)
    target_units = available_space if units is None else min(units, available_space)

    # Look up cached price + trade volume for this good
    cache = load_waypoint_cache()
    max_per_transaction = target_units  # Default: try to buy full amount
    cached_price = None
    if waypoint in cache:
        for good in cache[waypoint].get("trade_goods", []):
            if good.get("symbol") == trade_symbol:
                vol = good.get("tradeVolume")
                if vol and vol > 0:
                    max_per_transaction = vol
                cached_price = good.get("purchasePrice")
                break

    # Cap by affordability using cached price + current credits
    agent = _get_local_agent()
    credits = agent.get("credits")

    if credits is not None and cached_price and cached_price > 0:
        affordable = credits // cached_price
        if affordable <= 0:
            reason = (
                f"Insufficient credits ({credits} cr, need {cached_price} cr/unit)."
            )
            if min_qty:
                raise MinQtyNotMet(trade_symbol, 0, min_qty, reason)
            raise Exception(f"Cannot afford {trade_symbol}: {reason}")
        target_units = min(target_units, affordable)

    # Ensure dock
    _ensure_dock_logic(ship_symbol)

    # Make multiple purchases as needed
    total_purchased = 0
    total_cost = 0
    purchase_count = 0
    stop_reason = ""
    previous_price_per_unit = None

    while total_purchased < target_units:
        # Refresh local ship state implicitly via the interceptor!
        # The loop will see the updated cargo from the previous purchase.
        available_space = ship.cargo_capacity - ship.cargo_units

        if available_space <= 0:
            break  # Cargo full

        units_to_buy = min(
            max_per_transaction, available_space, target_units - total_purchased
        )

        data = client.buy_cargo(ship_symbol, trade_symbol, units_to_buy)
        _intercept(ship_symbol, data)

        if isinstance(data, dict) and "error" in data:
            if purchase_count == 0:
                raise Exception(data["error"])
            stop_reason = data["error"]
            break

        tx = data.get("transaction", {})
        units_bought = tx.get("units", units_to_buy)
        price_per_unit = tx.get("pricePerUnit", 0)

        total_purchased += units_bought
        total_cost += tx.get("totalPrice", 0)
        purchase_count += 1

        # Update cargo cost tracking
        ship.record_purchase(trade_symbol, units_bought, float(price_per_unit))

        # 1. Did we accidentally overpay on THIS transaction?
        if max_price and price_per_unit > max_price:
            stop_reason = f"Max price exceeded ({price_per_unit} > {max_price})"
            break

        # 2. The 1.25x Curve Prediction!
        if max_price and previous_price_per_unit is not None:
            price_delta = price_per_unit - previous_price_per_unit
            predicted_next_delta = price_delta * 1.25
            predicted_next_price = price_per_unit + predicted_next_delta

            if predicted_next_price > max_price:
                stop_reason = "Predicted to exceed max price"
                break

        previous_price_per_unit = price_per_unit

    # Check minimum quantity requirement
    if min_qty and total_purchased < min_qty:
        reason = (
            stop_reason
            or f"cargo space or credits limited purchase to {total_purchased}"
        )
        raise MinQtyNotMet(trade_symbol, total_purchased, min_qty, reason)

    # Final cargo state comes from local tracker (updated by interceptor)
    return f"Purchased {total_purchased} {trade_symbol} for {total_cost} cr. Cargo: {ship.cargo_units}/{ship.cargo_capacity}."


def _deliver_contract_logic(
    contract_id: str, ship_symbol: str, trade_symbol: str, units: int = None
) -> str:
    """
    Smart delivery: automatically calculates optimal units to deliver.
    Respects: contract requirements, cargo available, and explicit unit request.
    """
    # 1. Fetch contract to see remaining units needed
    contract = client.get_contract(contract_id)

    if not contract or contract.get("error", None):
        raise Exception(f"Error fetching contract {contract.get('error')}")

    if contract.get("fulfilled"):
        raise Exception(f"Contract {contract_id} is already fulfilled")

    # Find the delivery requirement for this trade symbol
    remaining_needed = 0
    for d in contract.get("terms", {}).get("deliver", []):
        if d.get("tradeSymbol") == trade_symbol:
            remaining_needed = d.get("unitsRequired", 0) - d.get("unitsFulfilled", 0)
            break

    if remaining_needed <= 0:
        raise Exception(f"Contract {contract_id} does not need any more {trade_symbol}")

    # 2. Check how much cargo is available
    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo: {cargo_data['error']}")

    available = 0
    for item in cargo_data.get("inventory", []):
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available <= 0:
        raise Exception(f"Ship {ship_symbol} has no {trade_symbol} to deliver")

    # 3. Calculate final delivery amount
    # Priority: contract remaining < cargo available < explicit units request
    if units is None:
        final_units = min(available, remaining_needed)
    else:
        final_units = min(units, available, remaining_needed)

    # 4. Deliver
    _ensure_dock_logic(ship_symbol)
    data = client.deliver_contract(contract_id, ship_symbol, trade_symbol, final_units)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    # Sync updated ship state to fleet tracker
    _intercept(ship_symbol, data)

    # Return summary
    contract = data.get("contract", {})
    terms = contract.get("terms", {})
    result = f"Delivered {final_units} {trade_symbol} to contract {contract_id}."

    # Check if all terms fulfilled
    all_fulfilled = True
    for d in terms.get("deliver", []):
        if d.get("unitsFulfilled", 0) < d.get("unitsRequired", 0):
            all_fulfilled = False
            break

    if all_fulfilled:
        # Auto-fulfill
        fulfill_data = client.fulfill_contract(contract_id)
        if isinstance(fulfill_data, dict) and "error" not in fulfill_data:
            result += " Contract FULFILLED!"
        else:
            result += f" (Fulfillment failed: {fulfill_data.get('error')})"
    else:
        for d in terms.get("deliver", []):
            if d.get("tradeSymbol") == trade_symbol:
                result += f" Progress: {d.get('unitsFulfilled', 0)}/{d.get('unitsRequired', 0)}"

    return result


def _refuel_ship_logic(ship_symbol: str) -> str:
    _ensure_dock_logic(ship_symbol)
    data = client.refuel(ship_symbol)
    _intercept(ship_symbol, data)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    fuel = data.get("fuel", {})
    tx = data.get("transaction", {})
    return f"Refueled {ship_symbol}. Fuel: {fuel.get('current')}/{fuel.get('capacity')}. Cost: {tx.get('totalPrice', '?')} cr."


def _buy_ship_logic(ship_type: str, waypoint_symbol: str, fleet=None) -> dict:
    """Core logic to purchase a ship at a shipyard.

    Returns dict with keys: ship, agent, credits_remaining
    Raises Exception on error.
    """
    data = client.purchase_ship(ship_type, waypoint_symbol)

    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    _intercept_agent(data)
    new_ship = data.get("ship", {})
    agent = data.get("agent", {})
    credits_remaining = agent.get("credits", 0)

    # Update fleet state with the new ship so it's instantly available
    if fleet and "ship" in data:
        fleet.update_from_api([data["ship"]])
    elif "ship" in data and (fleet := try_get_fleet()):
        get_fleet().update_from_api([data["ship"]])

    return {
        "ship": new_ship,
        "agent": agent,
        "credits_remaining": credits_remaining,
    }


def _transfer_cargo_logic(
    from_ship: str, to_ship: str, trade_symbol: str, units: int = None
) -> str:
    """Transfer cargo between ships. Auto-orbits both. Both must be at same waypoint.

    If trade_symbol is '*', transfers all cargo items.
    """
    # Check source ship inventory
    cargo_data = client.get_cargo(from_ship)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo for {from_ship}: {cargo_data['error']}")

    inventory = cargo_data.get("inventory", [])

    # Handle '*' to transfer all cargo
    if trade_symbol == "*":
        if not inventory:
            return f"No cargo to transfer from {from_ship}."

        # Ensure orbit states once
        _ensure_orbit_logic(from_ship)
        _ensure_orbit_logic(to_ship)

        results = []
        total_units = 0
        for item in inventory:
            symbol = item.get("symbol")
            available = item.get("units", 0)
            if available == 0:
                continue

            transfer_units = available if units is None else min(units, available)
            data = client.transfer_cargo(from_ship, to_ship, symbol, transfer_units)
            if isinstance(data, dict) and "error" in data:
                raise Exception(data["error"])

            # Sync updated cargo state to fleet tracker
            if isinstance(data, dict) and (fleet := try_get_fleet()):
                fleet.update_ship_partial(from_ship, data)

            results.append(f"  {symbol}: {transfer_units} units")
            total_units += transfer_units

        if not results:
            return f"No cargo available to transfer from {from_ship}."

        return (
            f"Transferred all cargo from {from_ship} to {to_ship}:\n"
            + "\n".join(results)
            + f"\nTotal: {total_units} units transferred"
        )

    # Handle single trade symbol
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        raise Exception(f"Ship {from_ship} has no {trade_symbol} available.")

    safe_units = available if units is None else min(units, available)

    # Ensure orbit states
    _ensure_orbit_logic(from_ship)
    _ensure_orbit_logic(to_ship)

    data = client.transfer_cargo(from_ship, to_ship, trade_symbol, safe_units)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])

    # Sync updated cargo state to fleet tracker
    if isinstance(data, dict) and (fleet := try_get_fleet()):
        fleet.update_ship_partial(from_ship, data)

    cargo = data.get("cargo", {})
    return (
        f"Transferred {safe_units} {trade_symbol} from {from_ship} to {to_ship}.\n"
        f"{from_ship} cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


# ──────────────────────────────────────────────
#  BEHAVIOR ENGINE
# ──────────────────────────────────────────────


class StepType(Enum):
    MINE = "mine"
    GOTO = "goto"
    BUY = "buy"
    SELL = "sell"
    DELIVER = "deliver"
    REFUEL = "refuel"
    SCOUT = "scout"
    CHART = "chart"
    TRANSFER = "transfer"
    SUPPLY = "supply"
    ALERT = "alert"
    REPEAT = "repeat"
    STOP = "stop"
    NEGOTIATE = "negotiate"
    BUY_SHIP = "buy_ship"
    AUTOTRADE = "autotrade"
    EXPLORE = "explore"
    CONSTRUCT = "construct"


@dataclass
class Step:
    step_type: StepType
    args: list[str] = field(default_factory=list)

    def __str__(self):
        if self.args:
            return f"{self.step_type.value} {' '.join(self.args)}"
        return self.step_type.value


def _refresh_waypoint_data(waypoint: str) -> None:
    """Fetch live market AND shipyard data for a waypoint, checking traits first. Silent on error."""
    system = "-".join(waypoint.split("-")[:2])
    cache = load_waypoint_cache()
    entry = cache.get(waypoint)

    # 1. If we don't know the traits, fetch the single waypoint first
    if not entry or "traits" not in entry:
        try:
            wp_data = client.get_waypoint(system, waypoint)
            if wp_data and isinstance(wp_data, dict) and "error" not in wp_data:
                _ingest_waypoints([wp_data])
                cache = load_waypoint_cache()
                entry = cache.get(waypoint, {})
            else:
                return  # Can't resolve waypoint
        except Exception:
            return

    # 2. Only fetch market if we know it has one
    if entry.get("has_market"):
        try:
            data = client.get_market(system, waypoint)
            if data and isinstance(data, dict) and "error" not in data:
                _save_market_to_cache(waypoint, data)
        except Exception:
            pass

    # 3. Only fetch shipyard if we know it has one
    if entry.get("has_shipyard"):
        try:
            data = client.get_shipyard(system, waypoint)
            if data and isinstance(data, dict) and "error" not in data:
                _save_shipyard_to_cache(waypoint, data)
        except Exception:
            pass


def parse_steps(steps_str: str) -> list[Step]:
    steps = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        verb = tokens[0].lower()
        args = tokens[1:]
        try:
            steps.append(Step(step_type=StepType(verb), args=args))
        except ValueError:
            raise ValueError(f"Unknown step type: '{verb}'")
    return steps


@dataclass
class BehaviorConfig:
    ship_symbol: str
    steps: list[Step]
    steps_str: str
    current_step_index: int = 0
    step_phase: str = "INIT"
    paused: bool = False
    error_message: str = ""
    alert_sent: bool = False
    last_action: str = ""  # Track most recent action for logging


_engine_instance: Optional["BehaviorEngine"] = None


def get_engine() -> "BehaviorEngine":
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BehaviorEngine()
    return _engine_instance


def _analyze_trade_routes(ship_symbol: str | None = None, min_profit: int = 1) -> list[dict]:
    """Helper: returns a list of trade dictionaries sorted by profitability."""
    import time

    cache = load_market_cache()
    if not cache:
        return []

    # Build per-good source/sink lists
    sources = {}  # good -> [(market_wp, buy_cost, volume)]
    sinks = {}  # good -> [(market_wp, sell_revenue, volume)]

    for wp, mdata in cache.items():
        if not isinstance(mdata, dict):
            continue
        trade_goods = mdata.get("trade_goods")
        if not trade_goods:
            continue

        # Identify exports/imports based on structural data + price availability
        exports = set(mdata.get("exports", []))
        imports = set(mdata.get("imports", []))
        exchange = set(mdata.get("exchange", []))
        source_goods = exports | exchange
        sink_goods = imports | exchange

        for tg in trade_goods:
            sym = tg["symbol"]
            buy_cost = tg.get("purchasePrice")
            sell_revenue = tg.get("sellPrice")

            if sym in source_goods and buy_cost is not None:
                sources.setdefault(sym, []).append(
                    (wp, buy_cost, tg.get("tradeVolume", 0))
                )
            if sym in sink_goods and sell_revenue is not None:
                sinks.setdefault(sym, []).append(
                    (wp, sell_revenue, tg.get("tradeVolume", 0))
                )

    all_goods = set(sources.keys()) & set(sinks.keys())
    if not all_goods:
        return []

    # Get ship position for distance calculation
    ship_pos = None
    wp_coords = {}
    if ship_symbol:
        try:
            ship_status = _get_local_ship(ship_symbol)
            ship_wp = ship_status.location or ""
            system_symbol = ship_wp.rsplit("-", 1)[0] if ship_wp else ""
            waypoints = get_system_waypoints(system_symbol)
            if isinstance(waypoints, list):
                for wp in waypoints:
                    wp_coords[wp["symbol"]] = (wp.get("x", 0), wp.get("y", 0))
                ship_pos = wp_coords.get(ship_wp)
        except Exception:
            pass

    routes = []
    now = time.time()

    for sym in all_goods:
        for src_wp, buy_cost, src_vol in sources.get(sym, []):
            for snk_wp, sell_rev, snk_vol in sinks.get(sym, []):
                if src_wp == snk_wp:
                    continue
                profit = sell_rev - buy_cost
                if profit < min_profit:
                    continue

                volume = min(src_vol, snk_vol)

                # Check staleness
                src_updated = cache.get(src_wp, {}).get("last_updated", 0)
                snk_updated = cache.get(snk_wp, {}).get("last_updated", 0)
                oldest = min(src_updated, snk_updated)
                stale = (now - oldest) > 7200 if oldest else True

                route = {
                    "good": sym,
                    "src": src_wp,
                    "snk": snk_wp,
                    "buy": buy_cost,
                    "sell": sell_rev,
                    "profit": profit,
                    "volume": volume,
                    "stale": stale,
                    "dist": None,
                    "ppm": 0,  # Profit per minute; 0 if no distance available
                }

                if ship_pos:
                    src_pos_coords = wp_coords.get(src_wp)
                    if src_pos_coords:
                        d = calculate_distance(
                            src_pos_coords[0], src_pos_coords[1],
                            ship_pos[0], ship_pos[1]
                        )
                        route["dist"] = max(d, 1.0)

                        # Calculate PPM: total profit for full cargo run divided by round-trip time
                        # Time estimate: 3 seconds per distance unit + 15s cooldown, doubled for round trip
                        travel_time_seconds = (route["dist"] * 3 + 15) * 2
                        travel_time_minutes = max(0.1, travel_time_seconds / 60)
                        total_profit = profit * volume
                        route["ppm"] = round(total_profit / travel_time_minutes, 2)

                routes.append(route)

    # Sort by PPM when available, fallback to profit
    routes.sort(
        key=lambda r: r.get("ppm", 0) if r.get("dist") else r["profit"],
        reverse=True
    )

    return routes


def _get_probe_plan(ship_symbol: str, ship_location: str, phase: int, claimed_targets: set = None, active_probe_systems: dict | None = None) -> str:
    """
    Determines a probe's mission using Time-Adjusted Staleness and Cluster Tours.

    1. Seed Selection: Uses (Age - FlightTime) to prioritize globally oldest markets.
    2. Cluster Tour: Once an urgent "Seed" is picked, adds nearby stale neighbors.
    """
    import time
    if active_probe_systems is None:
        active_probe_systems = {}
    system = ship_location.rsplit("-", 1)[0]
    cache = load_waypoint_cache()
    ship_status = _get_local_ship(ship_symbol)
    speed = max(1, ship_status.engine_speed)

    all_wps = [v for k, v in cache.items() if k != "_systems_fetched" and isinstance(v, dict)]
    system_wps = [wp for wp in all_wps if wp["symbol"].startswith(system + "-")]

    # Priority 1: Charting Local System
    uncharted = [wp for wp in system_wps if not wp.get("is_charted")]
    if uncharted:
        return "explore"

    # Priority 2: Market Refresh
    now = time.time()
    candidates = []

    sx, sy = cache.get(ship_location, {}).get("x", 0), cache.get(ship_location, {}).get("y", 0)

    # In phase 4, we consider ALL known markets. Otherwise, only local markets.
    search_wps = all_wps if phase >= 4 else system_wps
    #min_age = 3600 if phase >= 4 else 300  # 1 hour in phase 4, 5 mins in phase 1-3
    min_age = 600 if phase >= 4 else 300  # 1 hour in phase 4, 5 mins in phase 1-3

    for wp in search_wps:
        if wp.get("has_market"):
            wp_sym = wp["symbol"]
            if wp_sym == ship_location:
                continue
            if claimed_targets is not None and wp_sym in claimed_targets:
                continue

            last_updated = wp.get("last_updated", 0)
            if last_updated == 0:
                # Never scouted: cap age at 24h so the system clustering penalty actually works
                age = 86400
            else:
                age = now - last_updated

            if age < min_age:
                continue

            wx, wy = wp.get("x", 0), wp.get("y", 0)

            # Cross-system penalty
            if wp_sym.startswith(system + "-"):
                dist_from_ship = calculate_distance(sx, sy, wx, wy)
                flight_time = (dist_from_ship * (30 / speed)) + 15
            else:
                flight_time = 500  # Jump overhead penalty

            score = age - flight_time

            wp_sys = wp_sym.rsplit("-", 1)[0]
            if wp_sys != system:
                probes_in_target = active_probe_systems.get(wp_sys, 0)
                # Apply a ~5.5 hour score penalty per probe already in or headed to that system
                score -= (probes_in_target * 20000)

            candidates.append({"wp": wp_sym, "score": score, "age": age, "x": wx, "y": wy, "sys": wp_sys})

    if candidates:
        # 1. Pick the "Seed"
        candidates.sort(key=lambda x: x["score"], reverse=True)
        seed = candidates[0]

        # If seed is in another system, just go there and scout.
        if seed["sys"] != system:
            if claimed_targets is not None:
                claimed_targets.add(seed["wp"])
            return f"goto {seed['wp']}, scout, stop"

        tour = [seed]
        if claimed_targets is not None:
            claimed_targets.add(seed["wp"])

        # 2. Cluster Neighbors: Find up to 2 closest stale neighbors TO THE SEED.
        neighbors = [c for c in candidates if c["wp"] != seed["wp"] and c["sys"] == system and c["age"] > 600]
        neighbors.sort(key=lambda n: calculate_distance(seed["x"], seed["y"], n["x"], n["y"]))

        for n in neighbors[:2]:
            tour.append(n)
            if claimed_targets is not None:
                claimed_targets.add(n["wp"])

        # 3. Path Optimization: Sort the tour stops by distance from the SHIP.
        tour.sort(key=lambda t: calculate_distance(sx, sy, t["x"], t["y"]))

        steps = []
        for stop in tour:
            steps.append(f"goto {stop['wp']}")
            steps.append("scout")
        steps.append("stop")

        return ", ".join(steps)

    # Priority 3: Inter-System Expansion
    if phase >= 4:
        return "explore"

    # Failsafe
    return "stop"


def assign_idle_ships(fleet, engine):
    """The HQ Fleet Director. Automatically gives IDLE ships their next task."""
    if not get_hq_enabled():
        return

    idle_ships = engine.get_idle_ships(fleet)
    if not idle_ships:
        return

    strat = evaluate_fleet_strategy()
    targets_set = {t.strip() for t in _hq_managed_ships.split(",")}
    can_buy_ships = "ALL" in targets_set or "BUY_SHIPS" in targets_set

    needs_gate_materials = strat["needs_gate_materials"]

    # Get fleet activities: probe targets, active probe systems, and assignment flags
    acts = engine.get_fleet_activities(fleet)
    probe_targets = acts["targeted_waypoints"]
    active_probe_systems = acts["active_probe_systems"]
    constructor_assigned = len(acts["constructing_ships"]) > 0
    buyer_assigned = len(acts["buying_ships"]) > 0

    log.debug(f"👔 [HQ] Idle ships: {idle_ships} | Phase: {strat['phase']} | "
             f"Credits: {strat['credits']:,} | Excess: {strat['excess']:,} | "
             f"Can buy hauler: {strat['can_buy_ship']} | Needs probe: {strat['needs_probe']} "
             f"({strat['probe_count']}/{strat['desired_probes']}) | Shipyard: {strat['cheapest_shipyard']}")

    for ship_symbol in idle_ships:
        ship_status = fleet.get_ship(ship_symbol)
        if not ship_status or not ship_status.is_available():
            continue

        # Skip ships that aren't managed by HQ
        if not is_ship_hq_managed(ship_symbol, fleet):
            continue

        role = ship_status.role

        # FEATURE 2a: Buy Probe (cheap, high value — check first)
        if can_buy_ships and strat["needs_probe"] and not buyer_assigned and strat["cheapest_probe_shipyard"] != "Unknown":
            target_sy = strat["cheapest_probe_shipyard"]
            ship_to_buy = "SHIP_PROBE"

            if ship_status.location != target_sy:
                log.info(f"👔 [HQ] Dispatching {ship_symbol} to buy probe at {target_sy}.")
                engine.assign(ship_symbol, f"goto {target_sy}, buy_ship {ship_to_buy}, stop")
            else:
                log.info(f"👔 [HQ] {ship_symbol} at shipyard. Buying probe.")
                engine.assign(ship_symbol, f"buy_ship {ship_to_buy}, stop")

            buyer_assigned = True
            continue

        # FEATURE 2b: Buy Hauler (Can be done by any ship)
        if can_buy_ships and strat["can_buy_ship"] and not buyer_assigned and strat["cheapest_shipyard"] != "Unknown":
            target_sy = strat["cheapest_shipyard"]
            ship_to_buy = "SHIP_LIGHT_HAULER"

            if ship_status.location != target_sy:
                log.info(f"👔 [HQ] Dispatching {ship_symbol} to Shipyard at {target_sy} to buy {ship_to_buy}.")
                engine.assign(ship_symbol, f"goto {target_sy}, buy_ship {ship_to_buy}, stop")
            else:
                log.info(f"👔 [HQ] {ship_symbol} is at shipyard. Buying {ship_to_buy}.")
                engine.assign(ship_symbol, f"buy_ship {ship_to_buy}, stop")

            buyer_assigned = True
            continue

        # --- PROBES & SATELLITES ---
        if role == "SATELLITE":
            # Smart Scout (Refresh oldest market prices or expand)
            plan = _get_probe_plan(ship_symbol, ship_status.location, strat["phase"], claimed_targets=probe_targets, active_probe_systems=active_probe_systems)
            log.info(f"👔 [HQ] Dispatching {ship_symbol} to Scout: {plan}")
            engine.assign(ship_symbol, plan)

            # Update active probe systems so next idle probe this tick doesn't follow
            if plan.startswith("goto "):
                dest_wp = plan.split()[1].strip(",")
                dest_sys = get_system_from_waypoint(dest_wp)
                active_probe_systems[dest_sys] = active_probe_systems.get(dest_sys, 0) + 1
            elif plan == "explore":
                sys = get_system_from_waypoint(ship_status.location)
                active_probe_systems[sys] = active_probe_systems.get(sys, 0) + 1

            continue

        # --- HAULERS & COMMAND ---
        if role in ["HAULER", "COMMAND", "FREIGHTER"]:
            # FEATURE 1: Supply closest Jump Gate with needed materials
            # Assign first idle trader to construction; rest autotrade.
            # Avoid assigning multiple ships to construct by checking if any other ship is already doing it
            if needs_gate_materials and strat["excess"] > GATE_MIN_CREDIT_BUFFER and not constructor_assigned and not acts["supply_ships"]:
                log.info(f"👔 [HQ] Assigned {ship_symbol} to Jump Gate Construction.")
                engine.assign(ship_symbol, "construct")
                constructor_assigned = True
                continue

            # Default: Autotrade
            log.info(f"👔 [HQ] Assigned {ship_symbol} to Autotrade.")
            engine.assign(ship_symbol, "autotrade")


def _estimate_buyable_units(cached_price: int, max_price: int, trade_volume: int, spare_capacity: int) -> int:
    """Estimate how many units we can buy before price exceeds max_price.
    Uses a conservative 3% price increase per tradeVolume lot."""
    if cached_price <= 0 or trade_volume <= 0:
        return 0
    price = float(cached_price)
    total = 0
    for _ in range(20):  # safety cap
        if price > max_price:
            break
        lot = min(trade_volume, spare_capacity - total)
        if lot <= 0:
            break
        total += lot
        price = price * 1.03  # ~3% increase per lot
    return total


def _build_sell_sequence(ship_symbol: str, cargo_map: dict, ship) -> Optional[list[str]]:
    """
    Build a sequence of navigation and sell steps for unwanted cargo.

    Args:
        ship_symbol: Ship to sell cargo from
        cargo_map: Dict of {good_symbol: units}
        ship: Ship object with cargo_costs and location

    Returns:
        List of step strings (goto + sell commands) or None if can't find markets
    """
    if not cargo_map:
        return []

    pos = ship.location
    pending_sells = {}

    # Check if any cargo is needed for gate construction (special case)
    needs_gate = evaluate_fleet_strategy().get("needs_gate_materials", False)
    gate_mats = {"FAB_MATS", "ADVANCED_CIRCUITRY"}

    for good in cargo_map.keys():
        # If gate needs materials and we have them, prioritize gate supply over normal sell
        if needs_gate and good in gate_mats:
            jg_sym = get_engine()._find_closest_incomplete_jump_gate(pos)
            if jg_sym:
                return [f"goto {jg_sym}", f"supply {good}"]

        # Find best sell market
        best = _find_best_sell_market(ship_symbol, good)
        if best:
            cost = ship.cargo_costs.get(good, 0.0)
            if cost > 0:
                min_sell = int(cost * (1.0 + TRADE_PROFIT_MARGIN))
            else:
                min_sell = int(best["price"] * (1.0 - TRADE_PROFIT_MARGIN))
            pending_sells[good] = {"wp": best["wp"], "min_sell": min_sell}
        else:
            return None  # Can't find market for this cargo

    # Group sells by destination waypoint
    sell_groups = {}
    for good, info in pending_sells.items():
        sell_groups.setdefault(info["wp"], []).append((good, info["min_sell"]))

    # Build steps: goto each destination and sell
    steps = []
    for wp in sorted(sell_groups.keys()):
        steps.append(f"goto {wp}")
        for good, min_sell in sell_groups[wp]:
            steps.append(f"sell {good} min:{min_sell}")

    return steps


def _clear_unwanted_cargo_or_reassign(cfg, ship, next_step_name: str, engine) -> Optional[str]:
    """
    Helper: If ship has unwanted cargo, build a sell sequence and reassign behavior.

    Args:
        cfg: BehaviorConfig
        ship: Ship status object
        next_step_name: Step to append after selling (e.g., 'construct', 'negotiate')
        engine: BehaviorEngine instance

    Returns:
        None to continue step execution, or error string if cargo can't be sold
    """
    cargo = client.get_cargo(cfg.ship_symbol)
    inv = cargo.get("inventory", [])

    if cargo.get("units", 0) > 0:
        cargo_map = {item["symbol"]: item.get("units", 1) for item in inv}
        sell_steps = _build_sell_sequence(cfg.ship_symbol, cargo_map, ship)

        if sell_steps is None:
            cfg.paused = True
            cfg.alert_sent = True
            engine._save()
            return f"{cfg.ship_symbol} ALERT: Has cargo but can't find markets to sell."

        steps = sell_steps + [next_step_name]
        behavior_str = ", ".join(steps)
        engine.assign(cfg.ship_symbol, behavior_str)
        log.info(f"[{next_step_name.upper()}] {cfg.ship_symbol}: Has unwanted cargo. Auto-assigning: {behavior_str}")
        return None  # Step completed, new behavior assigned

    return None  # No cargo to clear, continue with normal step


def _plan_trade_route(ship_symbol: str, active_goods: set) -> str:
    """Build a multi-stop trade route that fills cargo across nearby markets."""
    ship = _get_local_ship(ship_symbol)
    pos = ship.location
    cargo_map = {item["symbol"]: item["units"] for item in ship.cargo_inventory}
    spare = ship.cargo_capacity - ship.cargo_units

    cache = load_waypoint_cache()

    steps = []
    pending_sells = {}
    goods_in_plan = set()

    def dist(a, b):
        if a not in cache or b not in cache:
            return 1000
        if get_system_from_waypoint(a) != get_system_from_waypoint(b):
            return 1000  # Penalize cross-system routing
        return waypoint_distance(a, b, cache)

    # Phase 1: Sell existing cargo
    if cargo_map:
        sell_steps = _build_sell_sequence(ship_symbol, cargo_map, ship)
        if sell_steps is None:
            return ""  # Cannot sell existing cargo, fail plan to trigger alert

        if sell_steps and sell_steps[0].startswith("goto "):
            # Gate supply special case: just return the gate sequence with autotrade
            return ", ".join(sell_steps) + ", autotrade"

        # Normal sell sequence: add to steps and track goods
        steps.extend(sell_steps)
        for good in cargo_map.keys():
            goods_in_plan.add(good)

    # Phase 2: Greedy buy loop
    routes = _analyze_trade_routes(ship_symbol, min_profit=15)
    routes = [r for r in routes if r["good"] not in active_goods]

    for _ in range(5):  # max 5 buy stops
        if spare <= 0:
            break

        candidates = []
        for r in routes:
            if r["good"] in goods_in_plan:
                continue
            # Skip routes where source and sink are the same market
            if r["src"] == r["snk"]:
                continue
            src_dist = dist(pos, r["src"])
            max_buy = int(r["buy"] * (1.0 + TRADE_PROFIT_MARGIN))
            units = _estimate_buyable_units(r["buy"], max_buy, r["volume"], spare)
            if units <= 0:
                continue
            total_profit = r["profit"] * units
            score = total_profit / max(1.0, src_dist)
            candidates.append((score, r, units))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, best, est_units = candidates[0]

        spare -= est_units
        log.info(f"[TradeRoute] {ship_symbol} pick: {best['good']} src={best['src']} snk={best['snk']} "
                 f"buy={best['buy']} sell={best['sell']} units={est_units} remaining_spare={spare} pos={pos}")

        # Sell-as-you-go: if best source is also a sell sink for pending cargo
        for good, sell_info in list(pending_sells.items()):
            d_sell = dist(pos, sell_info["wp"])
            d_src = dist(pos, best["src"])
            if sell_info["wp"] == best["src"] or d_sell < d_src:
                if pos != sell_info["wp"]:
                    steps.append(f"goto {sell_info['wp']}")
                    pos = sell_info["wp"]
                steps.append(f"sell {good} min:{sell_info['min_sell']}")
                del pending_sells[good]

        # Go to source and buy
        if pos != best["src"]:
            steps.append(f"goto {best['src']}")
            pos = best["src"]

        max_buy = int(best["buy"] * (1.0 + TRADE_PROFIT_MARGIN))
        steps.append(f"buy {best['good']} {est_units} max:{max_buy}")

        pending_sells[best["good"]] = {"wp": best["snk"], "min_sell": int(best["sell"] * (1.0 - TRADE_PROFIT_MARGIN))}
        goods_in_plan.add(best["good"])

    # Phase 3: Sell remaining (group by destination)
    sell_groups = {}
    for good, info in pending_sells.items():
        sell_groups.setdefault(info["wp"], []).append((good, info["min_sell"]))

    # Sort destinations by number of goods (sell more first) then distance
    for dest in sorted(sell_groups.keys(), key=lambda d: (-len(sell_groups[d]), dist(pos, d))):
        if pos != dest:
            steps.append(f"goto {dest}")
            pos = dest
        for good, min_sell in sell_groups[dest]:
            steps.append(f"sell {good} min:{min_sell}")

    return ", ".join(steps)


class BehaviorEngine:
    """
    Manages autonomous ship behaviors.
    Crucially, it uses the CORE LOGIC functions above, ensuring behavior ships
    are just as 'smart' (auto-refuel, etc) as LLM-controlled ships.
    """

    def __init__(self):
        self.behaviors: dict[str, BehaviorConfig] = {}
        self._last_mtime = 0.0
        self._load()

    def _load(self):
        if not BEHAVIORS_FILE.exists():
            return
        try:
            self._last_mtime = BEHAVIORS_FILE.stat().st_mtime
            entries = json.loads(BEHAVIORS_FILE.read_text())
            for e in entries:
                if "steps_str" in e:
                    try:
                        cfg = BehaviorConfig(
                            ship_symbol=e["ship_symbol"],
                            steps=parse_steps(e["steps_str"]),
                            steps_str=e["steps_str"],
                            current_step_index=e.get("current_step_index", 0),
                            step_phase=e.get("step_phase", "INIT"),
                            paused=e.get("paused", False),
                            error_message=e.get("error_message", ""),
                            alert_sent=e.get("alert_sent", False),
                            last_action=e.get("last_action", ""),
                        )
                        self.behaviors[cfg.ship_symbol] = cfg
                    except ValueError:
                        continue
        except Exception as exc:
            log.warning("Failed to load behaviors.json: %s", exc)

    def _save(self):
        data = [
            {
                "ship_symbol": c.ship_symbol,
                "steps_str": c.steps_str,
                "current_step_index": c.current_step_index,
                "step_phase": c.step_phase,
                "paused": c.paused,
                "error_message": c.error_message,
                "alert_sent": c.alert_sent,
                "last_action": c.last_action,
            }
            for c in self.behaviors.values()
        ]
        BEHAVIORS_FILE.write_text(json.dumps(data, indent=2))
        try:
            self._last_mtime = BEHAVIORS_FILE.stat().st_mtime
        except OSError:
            pass

    def sync_state(self):
        if not BEHAVIORS_FILE.exists():
            return
        try:
            if BEHAVIORS_FILE.stat().st_mtime > self._last_mtime:
                self.behaviors.clear()
                self._load()
        except OSError:
            pass

    def assign(self, ship_symbol: str, steps_str: str, start_step: int = 0) -> str:
        try:
            steps = parse_steps(steps_str)
        except ValueError as e:
            return f"Error: {e}"
        if not steps:
            return "Error: no steps provided"

        clamped = max(0, min(start_step, len(steps) - 1))
        self.behaviors[ship_symbol] = BehaviorConfig(
            ship_symbol=ship_symbol,
            steps=steps,
            steps_str=steps_str,
            current_step_index=clamped,
        )
        self._save()
        return f"Assigned to {ship_symbol}: {steps_str}"

    def cancel(self, ship_symbol: str) -> None:
        self.behaviors.pop(ship_symbol, None)
        self._save()

    def get_idle_ships(self, fleet) -> list[str]:
        return [s for s in fleet.ships if s not in self.behaviors]

    def get_fleet_activities(self, fleet=None, exclude_ship: str | None = None) -> dict:
        """Analyzes all active behaviors to summarize fleet activities.

        Returns:
            targeted_waypoints: set of waypoint symbols any ship has a GOTO step for
            active_probe_systems: dict of system_symbol -> probe count (probes in/heading to that system)
            active_goods: set of trade goods currently being bought by any ship (for autotrade dedup)
            constructing_ships: set of ship symbols with a CONSTRUCT step
            buying_ships: set of ship symbols with a BUY_SHIP step
            supply_ships: set of ship symbols with a SUPPLY step
        """
        activities = {
            "targeted_waypoints": set(),
            "active_probe_systems": {},
            "active_goods": set(),
            "constructing_ships": set(),
            "buying_ships": set(),
            "supply_ships": set(),
        }

        for sym, cfg in self.behaviors.items():
            if exclude_ship and sym == exclude_ship:
                continue

            ship_sys = None
            is_probe = False

            if fleet:
                ship_st = fleet.get_ship(sym)
                if ship_st:
                    if ship_st.location:
                        ship_sys = get_system_from_waypoint(ship_st.location)
                    if ship_st.role == "SATELLITE":
                        is_probe = True

            cross_sys_dests = set()
            for step in cfg.steps:
                if step.step_type == StepType.GOTO and step.args:
                    wp = step.args[0]
                    activities["targeted_waypoints"].add(wp)
                    if is_probe and ship_sys:
                        dest_sys = get_system_from_waypoint(wp)
                        if dest_sys != ship_sys:
                            cross_sys_dests.add(dest_sys)
                elif step.step_type == StepType.BUY and step.args:
                    activities["active_goods"].add(step.args[0])
                elif step.step_type == StepType.CONSTRUCT:
                    activities["constructing_ships"].add(sym)
                elif step.step_type == StepType.BUY_SHIP:
                    activities["buying_ships"].add(sym)
                elif step.step_type == StepType.SUPPLY:
                    activities["supply_ships"].add(sym)

            if is_probe and ship_sys:
                # If probe is heading to another system, count destination(s) only.
                # If probe is staying local, count current system.
                target_systems = cross_sys_dests if cross_sys_dests else {ship_sys}
                for sys in target_systems:
                    activities["active_probe_systems"][sys] = activities["active_probe_systems"].get(sys, 0) + 1

        return activities

    def summary(self) -> str:
        if not self.behaviors:
            return "(no behaviors assigned -- all ships idle)"
        lines = []
        for cfg in self.behaviors.values():
            step_idx = cfg.current_step_index
            current_step = cfg.steps[step_idx] if step_idx < len(cfg.steps) else "?"
            status = "PAUSED" if cfg.paused else cfg.step_phase
            if cfg.error_message:
                status = f"ERROR: {cfg.error_message}"
            lines.append(
                f"  {cfg.ship_symbol}: step {step_idx + 1}/{len(cfg.steps)} [{current_step}] ({status})"
            )
        return "\n".join(lines)

    def pause(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg:
            return f"{ship_symbol} has no assigned behavior."
        if cfg.paused:
            return f"{ship_symbol} is already paused."
        cfg.paused = True
        self._save()
        return f"Paused {ship_symbol} at step {cfg.current_step_index + 1}/{len(cfg.steps)}."

    def resume(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg or not cfg.paused:
            return "Nothing to resume."
        cfg.paused = False
        cfg.alert_sent = False
        #cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        self._save()
        return f"Resumed {ship_symbol}."

    def skip_step(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg:
            return "No behavior."
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        cfg.paused = False
        self._save()
        return f"Skipped step for {ship_symbol}."

    def tick(self, ship_symbol: str, fleet, client) -> Optional[str]:
        """
        Tick one behavior step for a ship.
        Returns alert string if needed (alert/error/pause condition).
        Returns None if step executed normally or ship unavailable.
        """
        cfg = self.behaviors.get(ship_symbol)
        if not cfg or cfg.paused:
            return None

        ship = fleet.get_ship(ship_symbol)
        if not ship or not ship.is_available():
            return None

        if cfg.current_step_index >= len(cfg.steps):
            cfg.current_step_index = 0
            cfg.step_phase = "INIT"

        step = cfg.steps[cfg.current_step_index]
        step_display = f"{step.step_type.value} {' '.join(step.args)}".strip()

        try:
            result = None
            if step.step_type == StepType.MINE:
                result = self._step_mine(cfg, step, ship, fleet)
            elif step.step_type == StepType.GOTO:
                result = self._step_goto(cfg, step, ship, fleet)
            elif step.step_type == StepType.BUY:
                result = self._step_buy(cfg, step, ship, fleet)
            elif step.step_type == StepType.SELL:
                result = self._step_sell(cfg, step, ship, fleet)
            elif step.step_type == StepType.REFUEL:
                result = self._step_refuel(cfg, step, ship, fleet)
            elif step.step_type == StepType.DELIVER:
                result = self._step_deliver(cfg, step, ship, fleet)
            elif step.step_type == StepType.TRANSFER:
                result = self._step_transfer(cfg, step, ship, fleet)
            elif step.step_type == StepType.SUPPLY:
                result = self._step_supply(cfg, step, ship, fleet)
            elif step.step_type == StepType.CHART:
                result = self._step_chart(cfg, step, ship, fleet)
            elif step.step_type == StepType.SCOUT:
                result = self._step_scout(cfg, step, ship, fleet)
            elif step.step_type == StepType.REPEAT:
                result = self._step_repeat(cfg, step, ship, fleet)
            elif step.step_type == StepType.STOP:
                result = self._step_stop(cfg)
            elif step.step_type == StepType.ALERT:
                result = self._step_alert(cfg, step)
            elif step.step_type == StepType.AUTOTRADE:
                result = self._step_autotrade(cfg)
            elif step.step_type == StepType.EXPLORE:
                result = self._step_explore(cfg, ship)
            elif step.step_type == StepType.CONSTRUCT:
                result = self._step_construct(cfg, ship, step)
            elif step.step_type == StepType.BUY_SHIP:
                result = self._step_buy_ship(cfg, step, ship, fleet)
            elif step.step_type == StepType.NEGOTIATE:
                result = self._step_negotiate(cfg)
            # Store last action for logging
            cfg.last_action = f"step {cfg.current_step_index + 1}: {step_display}"
            self._save()
            return result
        except Exception as e:
            cfg.error_message = str(e)
            cfg.paused = True
            cfg.last_action = (
                f"step {cfg.current_step_index + 1}: {step_display} [ERROR: {str(e)}]"
            )
            self._save()
            return f"{ship_symbol} ERROR: {e}"

    # ── Step Handlers using CORE LOGIC ───────────────────────────────────

    def _step_goto(self, cfg, step, ship, fleet) -> Optional[str]:
        """Smart navigation: Loops until ship reaches final destination. Auto-refuels."""
        dest_wp = step.args[0]
        # Optional mode argument: goto WAYPOINT [MODE]
        mode = step.args[1] if len(step.args) > 1 else "CRUISE"

        # 1. Check if we have arrived at the FINAL destination
        # We assume ship.location is accurate if we just refreshed (handled in WAITING phase)
        if ship.location == dest_wp and ship.nav_status != "IN_TRANSIT":
            self._advance(cfg, ship, fleet)
            return None

        # 2. INIT Phase: Prepare to move
        if cfg.step_phase == "INIT":
            # REMOVED: Redundant "Check Refuel Needs" block.
            # _navigate_ship_logic (called below) handles "Refuel at Departure" automatically.

            # Navigate to next hop
            try:
                msg, wait = _navigate_ship_logic(cfg.ship_symbol, dest_wp, mode=mode)

                if wait > 0:
                    fleet.set_transit(cfg.ship_symbol, wait)

                cfg.step_phase = "WAITING"
                self._save()
                return None
            except Exception as e:
                raise e

        # 3. WAITING Phase: Ship is moving
        if cfg.step_phase == "WAITING":
            if ship.is_available():
                # Timer expired. REFRESH STATE FROM API to confirm where we are.
                # This fixes the "hanging out at intermediate stop" bug.
                real_ship = client.get_ship(cfg.ship_symbol)
                if isinstance(real_ship, dict) and "error" not in real_ship:
                    fleet.update_from_api([real_ship])
                    # Update local ref since fleet updated the object in place or replaced it
                    ship = fleet.get_ship(cfg.ship_symbol)

                if ship.nav_status != "IN_TRANSIT":
                    # We have finished the transit.
                    if ship.location == dest_wp:
                        # Arrived at final destination
                        self._advance(cfg, ship, fleet)
                    else:
                        # Arrived at intermediate stop (or drift finished).
                        # Loop back to INIT to refuel and calculate next hop.
                        cfg.step_phase = "INIT"
                        self._save()
            return None

    def _step_mine(self, cfg, step, ship, fleet) -> Optional[str]:
        """Smart mining using _extract_ore_logic."""
        # Argument parsing with heuristic:
        # If arg[0] doesn't look like a waypoint (no dash), assume it's an ore and we mine at current location.
        asteroid_wp = None
        ore_types = []

        if step.args:
            first_arg = step.args[0]
            if "-" in first_arg:
                asteroid_wp = first_arg
                ore_types = step.args[1:]
            else:
                # First arg is likely an ore type (e.g. "COPPER_ORE")
                # Assume mine at current location
                asteroid_wp = ship.location
                ore_types = step.args

        if cfg.step_phase == "INIT":
            # If we need to go somewhere else to mine
            if asteroid_wp and ship.location != asteroid_wp:
                # Reuse navigate logic dynamically
                try:
                    msg, wait = _navigate_ship_logic(cfg.ship_symbol, asteroid_wp)
                    if wait > 0:
                        fleet.set_transit(cfg.ship_symbol, wait)
                    # We don't change phase to WAITING here because we want to loop back to INIT
                    # until we are actually at the location.
                    return None
                except Exception as e:
                    raise Exception(f"Nav error in mine step: {e}")

            cfg.step_phase = "EXTRACTING"
            self._save()
            return None

        if cfg.step_phase == "EXTRACTING":
            if ship.cargo_capacity > 0 and ship.cargo_units >= ship.cargo_capacity:
                # Alert operator when cargo is full — don't silently advance
                cfg.paused = True
                cfg.alert_sent = True
                self._save()
                return f"{cfg.ship_symbol} ALERT: Cargo full at {ship.location}. Define a sell/deliver/transfer step or manually empty cargo."

            msg, wait = _extract_ore_logic(cfg.ship_symbol)
            if wait > 0:
                fleet.set_extraction_cooldown(cfg.ship_symbol, wait)

            # Auto-Jettison Logic (simplified)
            if ore_types:
                # We need fresh cargo data to know what to trash
                c = client.get_cargo(cfg.ship_symbol)
                for item in c.get("inventory", []):
                    if item["symbol"] not in ore_types:
                        data = client.jettison(cfg.ship_symbol, item["symbol"], item["units"])
                        _intercept(cfg.ship_symbol, data)
            return None

    def _step_sell(self, cfg, step, ship, fleet) -> Optional[str]:
        target = step.args[0] if step.args else "*"
        # Parse min:PRICE from args
        min_price = None
        for arg in step.args[1:]:
            if arg.startswith("min:"):
                try:
                    parsed_min = int(arg[4:])
                    if parsed_min > 0:
                        min_price = parsed_min
                except ValueError:
                    pass
                break
        # Refresh market cache before selling so prices are current
        if ship.location:
            _refresh_waypoint_data(ship.location)
        # Get inventory
        c = client.get_cargo(cfg.ship_symbol)
        inventory = c.get("inventory", [])

        sold_something = False
        for item in inventory:
            sym = item["symbol"]
            if target == "*" or sym == target:
                try:
                    _sell_cargo_logic(cfg.ship_symbol, sym, min_price=min_price)
                    sold_something = True
                except PriceFloorHit as e:
                    cfg.paused = True
                    cfg.alert_sent = True
                    self._save()
                    return f"{cfg.ship_symbol} ALERT: {e}"
                except Exception as e:
                    # If target is *, ignore failures (e.g. contract goods)
                    if target != "*":
                        raise e

        self._advance(cfg, ship, fleet)
        return None

    def _step_buy(self, cfg, step, ship, fleet) -> Optional[str]:
        """Buy cargo from current market. Usage: buy TRADE_SYMBOL [UNITS] [max:PRICE] [min_qty:N]"""
        if not step.args:
            raise Exception("buy step requires trade symbol (e.g., 'buy IRON_ORE 10')")

        # Refresh market cache before buying so affordability check uses current prices
        if ship.location:
            _refresh_waypoint_data(ship.location)

        trade_symbol = step.args[0]
        units = None
        max_price = None
        min_qty = None
        for arg in step.args[1:]:
            if arg.startswith("max:"):
                max_price = int(arg[4:])
            elif arg.startswith("min_qty:"):
                min_qty = int(arg[8:])
            else:
                try:
                    units = int(arg)
                except ValueError:
                    pass

        try:
            _buy_cargo_logic(
                cfg.ship_symbol,
                trade_symbol,
                units,
                max_price=max_price,
                min_qty=min_qty,
            )
        except (PriceCeilingHit, MinQtyNotMet) as e:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: {e}"

        self._advance(cfg, ship, fleet)
        return None

    def _step_refuel(self, cfg, step, ship, fleet) -> Optional[str]:
        if ship.fuel_capacity > 0:
            _refuel_ship_logic(cfg.ship_symbol)
        self._advance(cfg, ship, fleet)
        return None

    def _step_deliver(self, cfg, step, ship, fleet) -> Optional[str]:
        """Deliver cargo for a contract. Usage: deliver CONTRACT_ID ITEM [UNITS]"""
        if len(step.args) < 2:
            raise Exception(
                "deliver step requires contract_id and trade_symbol (e.g., 'deliver CONT001 DIAMONDS 5')"
            )

        contract_id = step.args[0]
        trade_symbol = step.args[1]
        units = int(step.args[2]) if len(step.args) > 2 else None

        try:
            _deliver_contract_logic(contract_id, cfg.ship_symbol, trade_symbol, units)
        except Exception as e:
            raise e

        self._advance(cfg, ship, fleet)
        return None

    def _step_transfer(self, cfg, step, ship, fleet) -> Optional[str]:
        """Transfer cargo to another ship. Usage: transfer DESTINATION_SHIP TRADE_SYMBOL [UNITS]

        TRADE_SYMBOL can be '*' to transfer all cargo. UNITS is optional (defaults to max available).
        Examples: transfer SHIP-2 IRON_ORE, transfer SHIP-2 IRON_ORE 50, transfer SHIP-2 *
        """
        if len(step.args) < 2:
            raise Exception(
                "transfer step requires destination ship and trade symbol (e.g., 'transfer SHIP-2 IRON_ORE 50' or 'transfer SHIP-2 *')"
            )

        destination_ship = step.args[0]
        trade_symbol = step.args[1]
        units = int(step.args[2]) if len(step.args) > 2 else None

        try:
            _transfer_cargo_logic(
                cfg.ship_symbol, destination_ship, trade_symbol, units
            )
        except Exception as e:
            raise e

        self._advance(cfg, ship, fleet)
        return None

    def _step_supply(self, cfg, step, ship, fleet) -> Optional[str]:
        """Supply a construction project at current location. Usage: supply TRADE_SYMBOL [UNITS]

        TRADE_SYMBOL: cargo type to supply (FAB_MATS or ADVANCED_CIRCUITRY)
        UNITS: optional amount to supply. If omitted, supplies all cargo of that type.
        Examples: supply FAB_MATS, supply ADVANCED_CIRCUITRY 50
        """
        if len(step.args) < 1:
            raise Exception(
                "supply step requires trade symbol (e.g., 'supply FAB_MATS' or 'supply ADVANCED_CIRCUITRY 50')"
            )

        trade_symbol = step.args[0]
        units = int(step.args[1]) if len(step.args) > 1 else None

        # Ensure ship is docked
        _ensure_dock_logic(cfg.ship_symbol)

        # Use ship's current location as waypoint
        waypoint = ship.location
        if not waypoint:
            raise Exception(f"{cfg.ship_symbol} has no location")

        # Extract system from waypoint (e.g., "X1-ABC-123A" -> "X1-ABC")
        system = "-".join(waypoint.split("-")[:2])

        # If units not specified, use all cargo of that type
        if units is None:
            try:
                ship_status = _get_local_ship(cfg.ship_symbol)
                for item in ship_status.cargo_inventory:
                    if item.get("symbol") == trade_symbol:
                        units = item.get("units", 0)
                        break
                if units is None or units == 0:
                    raise Exception(f"No cargo of type {trade_symbol} available")
            except Exception as e:
                raise Exception(f"Could not get ship cargo data for {cfg.ship_symbol}: {e}")

        try:
            result = client.supply_construction(
                system, waypoint, cfg.ship_symbol, trade_symbol, units
            )
            _intercept(cfg.ship_symbol, result)
            if isinstance(result, dict) and "error" in result:
                raise Exception(f"Supply failed: {result['error']}")
        except Exception as e:
            raise e

        # Update cached construction status after supplying materials
        _fetch_and_cache_construction(waypoint)

        self._advance(cfg, ship, fleet)
        return None

    def _step_scout(self, cfg, step, ship, fleet) -> Optional[str]:
        if not ship.location:
            raise Exception("No location")
        _refresh_waypoint_data(ship.location)
        self._advance(cfg, ship, fleet)
        return None

    def _step_chart(self, cfg, step, ship, fleet) -> Optional[str]:
        if not ship.location:
            raise Exception("No location")

        # 1. Check unified cache to prevent double-charting
        cache = load_waypoint_cache()
        entry = cache.get(ship.location, {})

        if entry.get("is_charted"):
            self._advance(cfg, ship, fleet)
            return None

        # 2. Call API
        result = client.chart(cfg.ship_symbol)
        _intercept(cfg.ship_symbol, result)

        # 3. Handle errors (specifically 4230: Already Charted)
        if isinstance(result, dict) and "error" in result:
            err = result.get("error")
            if "already charted" in str(err).lower() or (isinstance(err, dict) and err.get("code") == 4230):
                if ship.location in cache:
                    cache[ship.location]["is_charted"] = True
                    _save_cache(cache)
                self._advance(cfg, ship, fleet)
                return None
            raise Exception(f"Chart error: {err}")

        # 4. Success - Ingest the newly charted waypoint data
        wp_data = result.get("waypoint")
        if wp_data:
            _ingest_waypoints([wp_data])

        self._advance(cfg, ship, fleet)
        return None

    def _step_alert(self, cfg, step) -> Optional[str]:
        if not cfg.alert_sent:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: {' '.join(step.args)}"
        return None

    def _step_autotrade(self, cfg) -> Optional[str]:
        """
        Dynamic Step:
        1. Analyzes market/cargo to generate a multi-stop trade route string.
        2. Calls self.assign() to overwrite the current behavior with the new plan.
        """
        # Find goods currently being bought by OTHER ships to avoid duplication
        active_goods = self.get_fleet_activities(exclude_ship=cfg.ship_symbol)["active_goods"]

        # 1. Check Cargo
        cargo_data = client.get_cargo(cfg.ship_symbol)
        _intercept(cfg.ship_symbol, cargo_data)

        # Error handling
        if isinstance(cargo_data, dict) and "error" in cargo_data:
            cfg.error_message = f"Error fetching cargo: {cargo_data['error']}"
            cfg.paused = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: {cargo_data['error']}"

        # Reject ships with 0 capacity
        if cargo_data.get("capacity", 0) == 0:
            cfg.error_message = f"Ship has 0 cargo capacity. Cannot autotrade."
            cfg.paused = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: Cannot autotrade—0 cargo capacity."

        # Generate multi-stop trade route
        plan = _plan_trade_route(cfg.ship_symbol, active_goods)

        if not plan:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: Auto-trade found no profitable routes or cannot sell existing cargo. Scout more markets or expand."

        new_plan = plan + ", stop"

        # 2. Delegate to assign()
        # This handles parsing, state reset (step 0, phase INIT), and saving automatically.
        # It completely replaces the current behavior config with the new one.
        self.assign(cfg.ship_symbol, new_plan)

        return None

    def _step_explore(self, cfg, ship) -> Optional[str]:
        """Auto-Explore a system. Finds un-charted waypoints and unscouted markets/shipyards."""
        if not ship.location:
            raise Exception("Ship has no known location.")

        sys_sym = ship.location.rsplit("-", 1)[0]
        wps = get_system_waypoints(sys_sym)
        if isinstance(wps, dict) and "error" in wps:
            raise Exception(wps["error"])

        cache = load_waypoint_cache()
        target_wp = None
        action = None

        acts = self.get_fleet_activities(exclude_ship=cfg.ship_symbol)
        targeted_wps = acts["targeted_waypoints"]
        targeted_systems = {get_system_from_waypoint(wp) for wp in targeted_wps}

        for wp in wps:
            wp_sym = wp["symbol"]

            # Skip if another ship is already en route here
            if wp_sym in targeted_wps:
                continue

            traits = [t["symbol"] for t in wp.get("traits", [])]

            # Priority 1: Charting
            if not wp.get("is_charted"):
                target_wp = wp_sym
                action = "chart"
                break

            # Priority 2: Scouting markets/shipyards
            if "MARKETPLACE" in traits or "SHIPYARD" in traits:
                if wp_sym not in cache or "last_updated" not in cache.get(wp_sym, {}):
                    target_wp = wp_sym
                    action = "scout"
                    break

        if target_wp:
            # Inject a goto and the action, then return to exploring
            self.assign(cfg.ship_symbol, f"goto {target_wp}, {action}, stop")
            return None

        # Priority 3: Jump to a new uncharted system
        fetched = cache.get("_systems_fetched", [])

        # 3a. Check LOCAL jump gate first
        jgs = [wp for wp in wps if wp.get("type") == "JUMP_GATE"]
        if jgs:
            jg_sym = jgs[0]["symbol"]
            jg_cache_data = cache.get(jg_sym, {})

            if "connections" not in jg_cache_data:
                jg_data = client.get_jump_gate(sys_sym, jg_sym)
                if isinstance(jg_data, dict) and "error" not in jg_data:
                    jg_cache_data["connections"] = jg_data.get("connections", [])
                    cache[jg_sym] = jg_cache_data
                    _save_cache(cache)

            connections = jg_cache_data.get("connections", [])
            for conn in connections:
                conn_sym = conn.get("symbol") if isinstance(conn, dict) else conn
                conn_sys = conn_sym.rsplit("-", 1)[0]
                if conn_sys not in fetched:
                    self.assign(cfg.ship_symbol, f"goto {conn_sym}, stop")
                    return None

        # 3b. Check ALL jump gates globally for unfetched connections
        all_wps = [v for k, v in cache.items() if k != "_systems_fetched" and isinstance(v, dict)]
        global_jgs = [wp for wp in all_wps if wp.get("type") == "JUMP_GATE"]
        for jg in global_jgs:
            jg_sys = jg["symbol"].rsplit("-", 1)[0]
            if jg_sys == sys_sym:
                continue # Already checked local

            jg_cache_data = cache.get(jg["symbol"], {})
            if "connections" not in jg_cache_data:
                jg_data = client.get_jump_gate(jg_sys, jg["symbol"])
                if isinstance(jg_data, dict) and "error" not in jg_data:
                    jg_cache_data["connections"] = jg_data.get("connections", [])
                    cache[jg["symbol"]] = jg_cache_data
                    _save_cache(cache)

            connections = jg_cache_data.get("connections", [])
            for conn in connections:
                conn_sym = conn.get("symbol") if isinstance(conn, dict) else conn
                conn_sys = conn_sym.rsplit("-", 1)[0]
                if conn_sys not in fetched:
                    self.assign(cfg.ship_symbol, f"goto {conn_sym}, stop")
                    return None

        # If no charting/scouting work or unfetched systems available:
        cfg.paused = True
        cfg.alert_sent = True
        self._save()
        return f"{cfg.ship_symbol} ALERT: Exploration complete! All known connected systems are fetched."

    def _find_closest_incomplete_jump_gate(self, ship_location: str) -> Optional[str]:
        """Find the closest incomplete JUMP_GATE from waypoint cache.

        Returns waypoint symbol of closest incomplete gate, or None if none found.
        """
        cache = load_waypoint_cache()
        ship_x, ship_y = None, None

        # Get ship coordinates from cache
        ship_wp = cache.get(ship_location, {})
        if isinstance(ship_wp, dict):
            ship_x = ship_wp.get("x")
            ship_y = ship_wp.get("y")

        candidates = []

        # Scan cache for JUMP_GATE waypoints
        for wp_sym, wp_data in cache.items():
            if not isinstance(wp_data, dict) or wp_sym == "_systems_fetched":
                continue

            # Check if it's a JUMP_GATE
            if wp_data.get("type") != "JUMP_GATE":
                continue

            # Check if incomplete (from cached construction status)
            const = wp_data.get("construction", {})
            if not const:
                continue  # No construction data cached yet
            if const.get("isComplete", False):
                continue

            # This gate is incomplete. Calculate distance if we have coords.
            dist = float("inf")
            if ship_x is not None and ship_y is not None:
                wp_x = wp_data.get("x")
                wp_y = wp_data.get("y")
                if wp_x is not None and wp_y is not None:
                    dist = calculate_distance(wp_x, wp_y, ship_x, ship_y)

            candidates.append((dist, wp_sym))

        if not candidates:
            return None

        # Return closest (sort by distance)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _step_construct(self, cfg, ship, step=None) -> Optional[str]:
        """Smart construction: Check jump gate needs, check budget, buy, supply.

        Optional: Provide waypoint as step argument (e.g., 'construct X1-PH29-K88').
        If no waypoint given, automatically finds the closest incomplete JUMP_GATE from cache.
        """
        if not ship.location:
            raise Exception("Ship has no known location.")

        # 1. Determine Target Jump Gate Waypoint
        # If step provides a waypoint argument, use it; otherwise find closest incomplete gate
        if step and step.args:
            jg_sym = step.args[0]
        else:
            jg_sym = self._find_closest_incomplete_jump_gate(ship.location)
            if not jg_sym:
                cfg.paused = True
                cfg.alert_sent = True
                self._save()
                return f"{cfg.ship_symbol} ALERT: No incomplete jump gates found in cache. Scout more systems."

        sys_sym = jg_sym.rsplit("-", 1)[0]

        # 2. Check Construction Status (from cache)
        cache = load_waypoint_cache()
        const = cache.get(jg_sym, {}).get("construction", {})
        if not const:
            # No cached data — fetch once and cache it
            _fetch_and_cache_construction(jg_sym)
            cache = load_waypoint_cache()
            const = cache.get(jg_sym, {}).get("construction", {})

        if const.get("isComplete", False) or not const.get("materials"):
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: Jump gate construction in {sys_sym} is COMPLETE!"

        # 3. Find needed material (pick least-completed one to alternate between materials)
        incomplete_materials = [m for m in const.get("materials", [])
                                if m.get("fulfilled", 0) < m.get("required", 0)]

        if not incomplete_materials:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return (
                f"{cfg.ship_symbol} ALERT: All materials fulfilled. Gate is complete!"
            )

        # Sort by completion percentage (ascending = least complete first)
        # This alternates supply runs between materials, giving source markets time to restock
        incomplete_materials.sort(
            key=lambda m: m.get("fulfilled", 0) / max(m.get("required", 1), 1)
        )

        mat = incomplete_materials[0]
        target_mat = mat["tradeSymbol"]
        needed_qty = mat["required"] - mat["fulfilled"]

        # 4. Cargo Check (If we already have the material, go supply it!)
        cargo = client.get_cargo(cfg.ship_symbol)
        _intercept(cfg.ship_symbol, cargo)
        inv = cargo.get("inventory", [])
        for item in inv:
            if item["symbol"] == target_mat:
                # We have some! Supply it.
                self.assign(
                    cfg.ship_symbol, f"goto {jg_sym}, supply {target_mat}, construct"
                )
                return None

        # Clear unwanted cargo if present
        clear_result = _clear_unwanted_cargo_or_reassign(cfg, ship, "construct", self)
        if clear_result is not None:
            return clear_result  # Either None (success) or error message

        # 4b. Check for other ships already delivering this material
        in_transit = 0
        acts = self.get_fleet_activities(exclude_ship=cfg.ship_symbol)
        for other_ship_sym in acts["constructing_ships"]:
            # Check this ship's cargo for the target material
            other_cargo = client.get_cargo(other_ship_sym)
            for item in other_cargo.get("inventory", []):
                if item["symbol"] == target_mat:
                    in_transit += item.get("units", 0)
                    log.debug(f"[Construct] {cfg.ship_symbol}: {other_ship_sym} has {item['units']} {target_mat} in transit")

        if in_transit > 0:
            needed_qty = max(0, needed_qty - in_transit)
            log.info(f"[Construct] {cfg.ship_symbol}: Reducing need from {needed_qty + in_transit} to {needed_qty} ({in_transit} already in transit)")

        if needed_qty <= 0:
            log.info(f"[Construct] {cfg.ship_symbol}: Enough material already in transit, going idle for HQ reassignment")
            return None  # Complete the construct step, let ship go idle

        # 5. Budget Check & Sourcing
        # Try to find source in current system first, then globally
        src_wp = _find_best_source(target_mat, sys_sym)
        if not src_wp:
            # Try searching globally across all cached systems
            global_results = _find_waypoints_logic(system_symbol="ALL", trade_symbol=target_mat)
            if global_results:
                # Pick cheapest available globally
                best = min(global_results, key=lambda r: r.get("market_match", {}).get("sellPrice", float("inf")))
                src_wp = best["symbol"]
            else:
                cfg.paused = True
                cfg.alert_sent = True
                self._save()
                return f"{cfg.ship_symbol} ALERT: Cannot find a market selling {target_mat} anywhere. Scout more systems."

        # Get price
        cache = load_market_cache()
        price = 5000  # default fallback
        for g in cache.get(src_wp, {}).get("trade_goods", []):
            if g["symbol"] == target_mat:
                price = g.get("purchasePrice", 5000)
                break

        capacity = ship.cargo_capacity
        buy_qty = min(capacity, needed_qty)
        cost = buy_qty * price

        # Verify Budget
        agent = _get_local_agent()
        credits = agent.get("credits", 0)

        # Calculate Required Reserve (same logic as the HUD)
        ships = list(get_fleet().ships.values()) if try_get_fleet() else []
        trader_count = sum(
            1
            for s in ships
            if s.role in ["COMMAND", "HAULER", "FREIGHTER"]
        )
        reserve_needed = max(trader_count * 100000, 100000)

        if credits < (reserve_needed + cost):
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: Need {cost:,}cr to buy {buy_qty} {target_mat}, but only have {credits:,}cr (keeping {reserve_needed:,}cr reserve). Paused to build capital."

        # 6. Execute!
        self.assign(
            cfg.ship_symbol,
            f"goto {src_wp}, buy {target_mat} {buy_qty} min_qty:1, goto {jg_sym}, supply {target_mat}, stop",
        )
        return None

    def _step_negotiate(self, cfg) -> Optional[str]:
        """
        Smart Contract Manager Step.
        1. If active contract exists -> Set behavior to fulfill it (Buy -> Deliver loop).
        2. If no active contract -> Go to HQ -> Negotiate -> Accept -> Loop.
        """
        # 0. Clear unwanted cargo if present
        ship = _get_local_ship(cfg.ship_symbol)
        clear_result = _clear_unwanted_cargo_or_reassign(cfg, ship, "negotiate", self)
        if clear_result is not None:
            return clear_result  # Either None (success) or error message

        # 1. Check for active unfulfilled contracts
        contracts = client.list_contracts()
        if isinstance(contracts, dict) and "error" in contracts:
            raise Exception(f"Failed to list contracts: {contracts['error']}")

        active_contract = next(
            (c for c in contracts if c.get("accepted") and not c.get("fulfilled")), None
        )

        if active_contract:
            # Plan route to fulfill it
            terms = active_contract.get("terms", {})
            # Find first incomplete delivery
            target_delivery = next(
                (
                    d
                    for d in terms.get("deliver", [])
                    if d.get("unitsFulfilled", 0) < d.get("unitsRequired", 0)
                ),
                None,
            )

            if not target_delivery:
                # Contract seems done but not fulfilled? Try fulfill.
                client.fulfill_contract(active_contract["id"])
                active_contract = None  # Proceed to get new one next loop
            else:
                c_id = active_contract["id"]
                symbol = target_delivery["tradeSymbol"]
                dest = target_delivery["destinationSymbol"]

                # Calculate exact quantity needed for this delivery
                units_needed = (
                    target_delivery["unitsRequired"] - target_delivery["unitsFulfilled"]
                )

                # Check current cargo
                cargo = client.get_cargo(cfg.ship_symbol)
                current_units = next(
                    (
                        i["units"]
                        for i in cargo.get("inventory", [])
                        if i["symbol"] == symbol
                    ),
                    0,
                )

                # Calculate how much to buy
                units_to_buy = max(0, units_needed - current_units)

                if units_to_buy == 0:
                    # Already have enough, just deliver
                    self.assign(
                        cfg.ship_symbol,
                        f"goto {dest}, deliver {c_id} {symbol}, negotiate",
                    )
                else:
                    # Need to buy more
                    ship_status = _get_local_ship(cfg.ship_symbol)
                    cargo_capacity = ship_status.cargo_capacity
                    units_in_cargo = ship_status.cargo_units

                    # Cap by available cargo space
                    units_to_buy = min(units_to_buy, cargo_capacity - units_in_cargo)

                    if units_to_buy <= 0:
                        # Cargo is full, deliver what we have first
                        self.assign(
                            cfg.ship_symbol,
                            f"goto {dest}, deliver {c_id} {symbol}, negotiate",
                        )
                    else:
                        ship_wp = ship_status.location or ""
                        sys_sym = ship_wp.rsplit("-", 1)[0] if ship_wp else ""
                        src = _find_best_source(symbol, sys_sym)
                        if not src:
                            cfg.paused = True
                            cfg.alert_sent = True
                            self._save()
                            return f"{cfg.ship_symbol} ALERT: Contract {c_id} needs {symbol} but no source found in cache."
                        self.assign(
                            cfg.ship_symbol,
                            f"goto {src}, buy {symbol} {units_to_buy}, goto {dest}, deliver {c_id} {symbol}, negotiate",
                        )
                return None

        # 2. No Active Contract -> Go to HQ and Negotiate
        ship_status = _get_local_ship(cfg.ship_symbol)
        hq = _get_local_agent().get("headquarters")

        if ship_status.location != hq:
            self.assign(cfg.ship_symbol, f"goto {hq}, negotiate")
            return None

        # At HQ: Negotiate & Accept
        _ensure_dock_logic(cfg.ship_symbol)
        data = client.negotiate_contract(cfg.ship_symbol)
        if isinstance(data, dict) and "error" in data:
            raise Exception(f"Negotiation failed: {data['error']}")
        client.accept_contract(data.get("contract", {}).get("id"))
        self.assign(cfg.ship_symbol, "negotiate")  # Loop to process the new contract
        return None

    def _step_repeat(self, cfg, step, ship, fleet) -> Optional[str]:
        if step.args:
            try:
                loops_left = int(step.args[0])
                if loops_left > 1:
                    # Decrement the counter
                    step.args[0] = str(loops_left - 1)
                    # Reconstruct the string so the new number persists to behaviors.json
                    cfg.steps_str = ", ".join(str(s) for s in cfg.steps)
                    cfg.current_step_index = 0
                    cfg.step_phase = "INIT"
                    self._save()
                    return None
                else:
                    # We've finished our last repeat. Move past this step.
                    self._advance(cfg, ship, fleet)
                    # If this was the final step in the sequence, stop the ship
                    # so it doesn't implicitly wrap around to 0 infinitely.
                    if cfg.current_step_index >= len(cfg.steps):
                        return self._step_stop(cfg)
                    return None
            except ValueError:
                pass  # Not a number, treat as infinite repeat

        # Infinite repeat (no valid N provided)
        cfg.current_step_index = 0
        cfg.step_phase = "INIT"
        self._save()
        return None

    def _step_stop(self, cfg) -> Optional[str]:
        """End the behavior and return ship to IDLE (manual control)."""
        self.cancel(cfg.ship_symbol)
        return None

    def _step_buy_ship(self, cfg, step, ship, fleet) -> Optional[str]:
        """Buy a ship at the current shipyard."""
        if not step.args:
            raise Exception("buy_ship requires a ship type (e.g., 'buy_ship SHIP_LIGHT_HAULER')")

        ship_type = step.args[0]
        waypoint = ship.location

        if not waypoint:
            raise Exception("Ship has no location")

        try:
            result = _buy_ship_logic(ship_type, waypoint, fleet=fleet)
            new_ship = result["ship"]
            new_ship_symbol = new_ship.get("symbol", "UNKNOWN")
            credits_remaining = result["credits_remaining"]
            log.info(f"🎉 [HQ] Purchased new ship: {new_ship_symbol} ({ship_type}) at {waypoint}. Credits remaining: {credits_remaining}")
            self._advance(cfg, ship, fleet)
            return None
        except Exception as e:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: Failed to purchase {ship_type} at {waypoint}: {str(e)}"

    def _evaluate_hq_opportunities(self, cfg, ship, fleet):
        """
        HQ JIT Planner Hook: Evaluate dynamic multi-cargo opportunities before advancing.

        This hook runs when a step completes and checks if the next step is a GOTO.
        If so, it calculates spare cargo capacity and dynamically inserts the single most
        profitable secondary cargo trade (profit > 15/unit) between current step and destination.
        On subsequent _advance calls, if more capacity exists, it will insert the next trade.
        """
        # Check if the next step exists and is a GOTO
        next_idx = cfg.current_step_index + 1
        if next_idx >= len(cfg.steps):
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: No next step after current index")
            return  # No next step or behavior will loop

        next_step = cfg.steps[next_idx]
        if next_step.step_type != StepType.GOTO:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: Next step is not GOTO, it's {next_step.step_type}")
            return  # Not a navigation step

        dest_waypoint = next_step.args[0] if next_step.args else None
        if not dest_waypoint or not ship.location:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: Missing destination or ship location")
            return  # Can't determine route

        if ship.location == dest_waypoint:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: Already at destination {dest_waypoint}")
            return  # Already at destination

        # Calculate spare cargo capacity
        spare_capacity = ship.cargo_capacity - ship.cargo_units
        if spare_capacity <= 0:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: No spare cargo capacity ({ship.cargo_units}/{ship.cargo_capacity})")
            return  # No room for additional cargo

        log.info(f"[HQ JIT] {cfg.ship_symbol}: Checking opportunities {ship.location} -> {dest_waypoint} (spare: {spare_capacity} units)")

        # Load market cache for both current and destination waypoints
        cache = load_waypoint_cache()
        current_entry = cache.get(ship.location, {})
        dest_entry = cache.get(dest_waypoint, {})

        if not current_entry.get("has_market") or not dest_entry.get("has_market"):
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: Missing market data at {ship.location} or {dest_waypoint}")
            return  # Can't trade without markets at both locations

        # Get cached trade_goods for current and destination waypoints
        current_goods = current_entry.get("trade_goods", [])
        dest_goods = dest_entry.get("trade_goods", [])

        if not current_goods or not dest_goods:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: No trade goods cached for route")
            return  # No market data cached

        # Build lookup dicts by symbol
        current_exchange = {item["symbol"]: item for item in current_goods}
        dest_exchange = {item["symbol"]: item for item in dest_goods}

        # Build set of goods already in the behavior sequence
        goods_in_sequence = set()
        for step in cfg.steps:
            if step.step_type == StepType.BUY and step.args:
                goods_in_sequence.add(step.args[0])
            elif step.step_type == StepType.SELL and step.args:
                goods_in_sequence.add(step.args[0])

        # Find profitable secondary cargo trades
        profitable_trades = []
        for symbol, item in current_exchange.items():
            if symbol in goods_in_sequence:
                continue  # Skip goods already in sequence

            buy_price = item.get("purchasePrice", 0)
            if symbol not in dest_exchange:
                continue  # Can't sell at destination

            sell_price = dest_exchange[symbol].get("sellPrice", 0)
            profit_per_unit = sell_price - buy_price

            if profit_per_unit > 15:  # Profit threshold: > 15 credits per unit
                profitable_trades.append({
                    "symbol": symbol,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "profit_per_unit": profit_per_unit,
                    "trade_volume": item.get("tradeVolume", 0),
                })

        if not profitable_trades:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: No profitable trades found (threshold: >15 credits/unit)")
            return  # No profitable opportunities

        # Sort by profit per unit (descending)
        profitable_trades.sort(key=lambda x: x["profit_per_unit"], reverse=True)

        # Process only the single most profitable trade
        trade = profitable_trades[0]
        symbol = trade["symbol"]
        buy_price = trade["buy_price"]
        sell_price = trade["sell_price"]
        profit_per_unit = trade["profit_per_unit"]

        units = min(spare_capacity, trade["trade_volume"])
        if units <= 0:
            log.debug(f"[HQ JIT] {cfg.ship_symbol}: Can't fit {symbol} (need {trade['trade_volume']}, have {spare_capacity})")
            return

        # Set safety margin
        max_buy = int(buy_price * (1.0 + JIT_TRADE_MARGIN))
        min_sell = int(sell_price * (1.0 - JIT_TRADE_MARGIN))

        # Create BUY and SELL steps
        buy_step = Step(StepType.BUY, [symbol, str(units), f"max:{max_buy}"])
        sell_step = Step(StepType.SELL, [symbol, f"min:{min_sell}"])

        # Insert BUY before the GOTO, SELL after the GOTO
        # Result: buy HERE, goto DEST, sell THERE
        buy_insert = next_idx
        sell_insert = next_idx + 1  # After the GOTO step

        cfg.steps.insert(buy_insert, buy_step)
        log.info(f"[HQ JIT] {cfg.ship_symbol}: Inserted BUY step: {buy_step}")

        sell_insert += 1  # Adjust for the BUY we just inserted
        cfg.steps.insert(sell_insert, sell_step)
        log.info(f"[HQ JIT] {cfg.ship_symbol}: Inserted SELL step: {sell_step}")

        # Rebuild steps_str to persist changes
        cfg.steps_str = ", ".join(str(s) for s in cfg.steps)
        self._save()

        # Summary log message
        log.info(f"👔 [HQ JIT Planner] {cfg.ship_symbol}: {symbol} x{units} @ {buy_price}cr (profit: {profit_per_unit}cr/unit). Next cycle will check for more.")

    def _advance(self, cfg, ship, fleet):
        """
        Advance to the next step in the behavior sequence.
        Also evaluates HQ JIT opportunities for dynamic multi-cargo packing.
        """
        # HQ JIT Planner Hook: Check for dynamic cargo opportunities before advancing
        self._evaluate_hq_opportunities(cfg, ship, fleet)

        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        cfg.error_message = ""
        cfg.alert_sent = False
        self._save()


# ──────────────────────────────────────────────
#  Observation tools
# ──────────────────────────────────────────────


@tool
def list_alerts() -> str:
    """[READ-ONLY] List all active alerts from the behavior engine."""
    if not _alert_queue:
        return "No active alerts."

    lines = ["Active Alerts:"]
    for i, alert in enumerate(_alert_queue):
        lines.append(f"  [{i}] {alert}")
    return "\n".join(lines)


@tool
def clear_alert(index: int) -> str:
    """[STATE: alerts] Clear an alert by its index number. Use list_alerts to see index numbers."""
    try:
        msg = _alert_queue.pop(index)
        return f"Cleared alert {index}: {msg}"
    except IndexError:
        return f"Error: No alert found at index {index}. Run list_alerts to see valid indices."


@tool
def view_agent() -> str:
    """[READ-ONLY] View your agent's credits, headquarters location, and ship count."""
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
def view_advisor(system_symbol: str = "") -> str:
    """[READ-ONLY] View the Fleet Strategy, Budget, and HQ Director status."""
    sys_sym = system_symbol if system_symbol else None
    return get_financial_assessment(sys_sym)


@tool
def view_contracts() -> str:
    """[READ-ONLY] List active contracts with status, terms, and delivery requirements."""
    data = client.list_contracts()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return "No contracts available."

    # Filter out fulfilled contracts to prevent UI/LLM text spew
    active_contracts = [c for c in data if not c.get("fulfilled")]
    fulfilled_count = len(data) - len(active_contracts)

    lines = []
    if fulfilled_count > 0:
        lines.append(
            f"(Hiding {fulfilled_count} historically fulfilled contracts. Showing active only.)\n"
        )

    if not active_contracts:
        lines.append("No active unfulfilled contracts at this time.")
        return "\n".join(lines)

    for c in active_contracts:
        lines.append(f"Contract: {c['id']}")
        lines.append(
            f"  Type: {c['type']}  |  Accepted: {c['accepted']}  |  Fulfilled: {c['fulfilled']}"
        )
        terms = c.get("terms", {})
        lines.append(
            f"  Payment: {terms.get('payment', {}).get('onAccepted', 0)} on accept, "
            f"{terms.get('payment', {}).get('onFulfilled', 0)} on fulfill"
        )
        for d in terms.get("deliver", []):
            lines.append(
                f"  Deliver: {d['unitsRequired']} {d['tradeSymbol']} to {d['destinationSymbol']} "
                f"({d['unitsFulfilled']}/{d['unitsRequired']} done)"
            )
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
def view_ships(system_symbol: str | None = None) -> str:
    """[READ-ONLY] List all ships with location, fuel, status, and assigned behaviors.

    Args:
        system_symbol: Optional. Filter to only show ships in this system (e.g. 'X1-KD26').
    """
    data = client.list_ships()
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    if (fleet := try_get_fleet()):
        fleet.update_from_api(data)

    # Filter by system first if requested
    if system_symbol:
        data = [s for s in data if s.get("nav", {}).get("systemSymbol") == system_symbol]

    engine = get_engine()

    # Group ships by system
    systems = {}
    for s in data:
        sys_sym = s.get("nav", {}).get("systemSymbol", "UNKNOWN")
        systems.setdefault(sys_sym, []).append(s)

    lines = []

    for sys, ships in sorted(systems.items()):
        lines.append(f"\n=== System: {sys} ({len(ships)} ships) ===")

        for s in ships:
            symbol = s["symbol"]
            nav = s.get("nav", {})
            fuel = s.get("fuel", {})
            cargo = s.get("cargo", {})

            # Check extraction cooldown
            cd_text = ""
            if (fleet := try_get_fleet()):
                ship_status = fleet.get_ship(symbol)
                if (
                    ship_status
                    and not ship_status.is_available()
                    and ship_status.busy_reason == "extraction_cooldown"
                ):
                    rem = ship_status.seconds_until_available()
                    cd_text = f" [COOLDOWN: {rem:.0f}s]"

            # Check Transit
            status = nav.get("status", "?")
            arrival_text = ""
            if status == "IN_TRANSIT":
                arrival_seconds = _parse_arrival(nav)
                if arrival_seconds > 0:
                    arrival_text = f" (Arriving in {int(arrival_seconds)}s)"

            # Check Behavior Status
            beh_text = ""
            cfg = engine.behaviors.get(symbol)
            if cfg:
                state = "PAUSED" if cfg.paused else cfg.step_phase
                if cfg.error_message:
                    state = f"ERR: {cfg.error_message}"
                # Highlight current step with [[ ]]
                steps = cfg.steps_str.split(",")
                if cfg.current_step_index < len(steps):
                    steps[cfg.current_step_index] = f"[[ {steps[cfg.current_step_index].strip()} ]]"
                highlighted_steps = ", ".join(steps)
                beh_text = f" | Job: {highlighted_steps} [{state}]"

            lines.append(f"   {symbol} ({s.get('registration', {}).get('role', '?')})")
            lines.append(
                f"   Loc: {nav.get('waypointSymbol', '?')} ({status}){arrival_text}"
            )
            lines.append(
                f"   Fuel: {fuel.get('current', 0)}/{fuel.get('capacity', 0)} | Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', 0)}{cd_text}{beh_text}"
            )
            lines.append("")

    if not lines:
        return (
            f"No ships found in {system_symbol}."
            if system_symbol
            else "No ships found."
        )

    return "\n".join(lines).strip()


@tool
def view_shipyards() -> str:
    """[READ-ONLY] List all known shipyards with available ship types and prices.

    Searches all cached systems for SHIPYARD waypoints. Shows what ships are available
    for purchase at each yard, along with prices from cache when available.
    """
    # Find all shipyards across cached systems
    results = _find_waypoints_logic(system_symbol="ALL", trait="SHIPYARD")

    if not results:
        return "No shipyards found in cache. Scout more systems to discover shipyards."

    cache = load_waypoint_cache()
    lines = []
    shipyard_count = 0

    for result in results:
        wp_sym = result["symbol"]
        sys_sym = wp_sym.rsplit("-", 1)[0]
        wp_data = cache.get(wp_sym, {})

        # Get shipyard data from cache
        ships = wp_data.get("ships", [])
        if not ships:
            # No shipyard data cached yet
            lines.append(f"\n   {wp_sym} — (no ship data cached)")
            shipyard_count += 1
            continue

        shipyard_count += 1
        lines.append(f"\n=== Shipyard: {wp_sym} ===")

        for ship in ships:
            ship_type = ship.get("type", "UNKNOWN")
            price = ship.get("purchasePrice", "?")
            price_str = f"{price:,}cr" if isinstance(price, int) else str(price)
            lines.append(f"   • {ship_type}: {price_str}")

    if not lines:
        return "All shipyards found but none have cached data yet. Scout shipyards to populate cache."

    header = f"\n=== {shipyard_count} Shipyards Found ===\n"
    return header + "\n".join(lines).strip()


def _buys_or_sells(m_data: dict, target: str):
    # Define logic for buying and selling
    buys = target in (m_data.get("imports", []) + m_data.get("exchange", []))
    sells = target in (m_data.get("exports", []) + m_data.get("exchange", []))
    hit = False
    match_reason = ""

    if buys or sells:
        hit = True
        actions = []
        if buys:
            actions.append("buys")
        if sells:
            actions.append("sells")

        # Joins with "and/or" if both are true, otherwise just the single action
        action_str = " and ".join(actions)
        match_reason = f"market {action_str} {target}"

    return hit, match_reason



def _find_best_sell_market(ship_symbol: str, good: str) -> Optional[dict]:
    """Helper: finds the best market to sell existing cargo."""
    try:
        ship_status = _get_local_ship(ship_symbol)
        ship_wp = ship_status.location or ""
        if not ship_wp:
            return None
        system = ship_wp.rsplit("-", 1)[0]
    except Exception:
        return None

    # Use shared logic - searching for the good
    results = _find_waypoints_logic(system_symbol=system, trade_symbol=good)

    candidates = []
    for res in results:
        mm = res["market_match"]
        # purchasePrice = what the market pays us (Agent Sell Price)
        # sellPrice = what the market charges us (Agent Buy Price)
        # For selling cargo, we want purchasePrice (what they pay us)
        if mm["purchasePrice"]:
            candidates.append({"wp": res["symbol"], "price": mm["purchasePrice"]})

    if not candidates:
        return None

    # Sort by price descending (Best Sell Price)
    candidates.sort(key=lambda x: x["price"], reverse=True)
    return candidates[0]


def _find_best_source(trade_symbol: str, ship_system: str) -> Optional[str]:
    """Find cheapest/best market for a good in the system using cache."""
    # Use shared logic
    results = _find_waypoints_logic(
        system_symbol=ship_system, trade_symbol=trade_symbol
    )

    candidates = []
    for res in results:
        mm = res["market_match"]
        # We want to BUY, so we look for market's sellPrice (Agent Buy Price)
        price = float("inf")
        valid = False

        if mm["sellPrice"]:
            price = mm["sellPrice"]
            valid = True
        elif "Export" in mm["roles"] or "Exchange" in mm["roles"]:
            # It's sold here, but no price in cache. Penalty applied.
            price = 1000000
            valid = True

        if valid:
            candidates.append((price, res["symbol"]))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])  # Lowest price first
    return candidates[0][1]


def _find_waypoints_logic(
    system_symbol: str,
    waypoint_type: str | None = None,
    trait: str | None = None,
    trade_symbol: str | None = None,
    ref_coords: tuple | None = None,
) -> list[dict]:
    """
    Core logic for finding waypoints. Returns a list of dicts containing:
    {
        'symbol': str,
        'distance': float (inf if no ref_coords),
        'waypoint': dict (raw API data),
        'market_match': dict (optional, contains price/role info for trade_symbol)
    }

    Args:
        system_symbol: System to search (e.g. 'X1-KD26'). Use 'ALL' to search globally across all cached systems.
        waypoint_type: Optional filter by type (e.g. 'ASTEROID', 'JUMP_GATE').
        trait: Optional filter by trait (e.g. 'MARKETPLACE', 'SHIPYARD').
        trade_symbol: Optional search for a specific trade good (e.g. 'IRON_ORE').
        ref_coords: Optional (x, y) tuple for distance calculation.
    """
    candidates = []

    # 1. Fetch Structural Data (needed for coords and Type/Trait filtering)
    # Support "ALL" to search across all cached systems
    cache = load_waypoint_cache()

    if system_symbol == "ALL":
        # Grab all waypoints from cache that have been fetched
        fetched_systems = cache.get("_systems_fetched", [])
        if not fetched_systems:
            return []  # No systems cached yet

        all_wps = []
        for k, v in cache.items():
            if k == "_systems_fetched" or not isinstance(v, dict):
                continue
            all_wps.append(v)

        wp_lut = {w["symbol"]: w for w in all_wps}
    else:
        # Fetch single system waypoints
        all_wps = get_system_waypoints(system_symbol)
        if isinstance(all_wps, dict) and "error" in all_wps:
            return []  # Or raise error

        wp_lut = {w["symbol"]: w for w in all_wps} if isinstance(all_wps, list) else {}

    # 2. Logic Branch: Trade Symbol Search
    if trade_symbol:
        for wp_sym, mdata in cache.items():
            if not isinstance(mdata, dict):
                continue
            # If searching a specific system, filter by system prefix
            if system_symbol != "ALL" and not wp_sym.startswith(system_symbol):
                continue

            # Check if good exists here
            goods = mdata.get("trade_goods", [])
            exports = mdata.get("exports", [])
            imports = mdata.get("imports", [])
            exchange = mdata.get("exchange", [])

            price_match = next((g for g in goods if g["symbol"] == trade_symbol), None)

            is_import = trade_symbol in imports
            is_export = trade_symbol in exports
            is_exchange = trade_symbol in exchange

            if price_match or is_import or is_export or is_exchange:
                # Calculate distance
                dist = float("inf")
                wp_data = wp_lut.get(wp_sym, {})

                if ref_coords and wp_data:
                    wx, wy = wp_data.get("x"), wp_data.get("y")
                    if wx is not None:
                        dist = calculate_distance(
                            wx, wy, ref_coords[0], ref_coords[1]
                        )

                candidates.append(
                    {
                        "symbol": wp_sym,
                        "distance": dist,
                        "waypoint": wp_data,
                        "market_match": {
                            "purchasePrice": (
                                price_match.get("purchasePrice")
                                if price_match
                                else None
                            ),
                            "sellPrice": (
                                price_match.get("sellPrice") if price_match else None
                            ),
                            "volume": (
                                price_match.get("tradeVolume") if price_match else None
                            ),
                            "roles": [
                                k
                                for k, v in [
                                    ("Import", is_import),
                                    ("Export", is_export),
                                    ("Exchange", is_exchange),
                                ]
                                if v
                            ],
                        },
                    }
                )

    # 3. Logic Branch: Waypoint/Trait Search (only if trade_symbol is not set)
    else:
        # Filter the structure data directly
        filtered_wps = []

        # Handle the ASTEROID special case logic from original tool
        target_types = [waypoint_type] if waypoint_type else []
        if waypoint_type == "ASTEROID":
            target_types.append("ENGINEERED_ASTEROID")

        for wp in all_wps:
            # Type Filter
            if target_types and wp["type"] not in target_types:
                continue

            # Trait Filter
            wp_traits = [t["symbol"] for t in wp.get("traits", [])]
            if trait and trait not in wp_traits:
                continue

            filtered_wps.append(wp)

        for wp in filtered_wps:
            dist = float("inf")
            if ref_coords:
                dist = calculate_distance(
                    wp["x"], wp["y"], ref_coords[0], ref_coords[1]
                )

            candidates.append(
                {
                    "symbol": wp["symbol"],
                    "distance": dist,
                    "waypoint": wp,
                    "market_match": None,
                }
            )

    # Global Sort by distance
    candidates.sort(key=lambda x: x["distance"])

    return candidates


@tool
def view_cargo(ship_symbol: str) -> str:
    """[READ-ONLY] View the cargo contents of a specific ship with costs."""
    data = client.get_cargo(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    inventory = data.get("inventory", [])
    capacity = data.get("capacity", "?")
    units = data.get("units", 0)
    lines = [f"Cargo for {ship_symbol}: {units}/{capacity} units"]
    if not inventory:
        lines.append("  (empty)")
    else:
        # Get cargo costs from FleetTracker if available
        ship = _get_local_ship(ship_symbol) if try_get_fleet() else None
        for item in inventory:
            symbol = item['symbol']
            item_units = item['units']
            cost = ship.cargo_costs.get(symbol, 0.0) if ship else 0.0
            if cost > 0:
                total_value = int(cost * item_units)
                lines.append(f"  {symbol}: {item_units} units @ {cost:.0f}cr ea = {total_value:,}cr total")
            else:
                lines.append(f"  {symbol}: {item_units} units")
    return "\n".join(lines)


@tool
def view_ship_details(ship_symbol: str) -> str:
    """[READ-ONLY] View detailed ship info: mounts, modules, capabilities."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        return f"Error: {ship['error']}"

    lines = [f"=== {ship_symbol} Details ==="]

    # Basic info
    reg = ship.get("registration", {})
    lines.append(f"Role: {reg.get('role', '?')}")
    lines.append(f"Faction: {reg.get('factionSymbol', '?')}")

    # Frame
    frame = ship.get("frame", {})
    lines.append(f"\nFrame: {frame.get('name', '?')} ({frame.get('symbol', '?')})")
    lines.append(
        f"  Module slots: {frame.get('moduleSlots', '?')}, Mounting points: {frame.get('mountingPoints', '?')}"
    )

    # Engine
    engine = ship.get("engine", {})
    lines.append(
        f"\nEngine: {engine.get('name', '?')} (speed: {engine.get('speed', '?')})"
    )

    # Mounts (weapons, lasers, etc.)
    mounts = ship.get("mounts", [])
    if mounts:
        lines.append(f"\nMounts ({len(mounts)}):")
        for m in mounts:
            lines.append(f"  • {m.get('name', m.get('symbol', '?'))}")
            if m.get("strength"):
                lines.append(f"    Strength: {m.get('strength')}")
    else:
        lines.append("\nMounts: None")

    # Modules
    modules = ship.get("modules", [])
    if modules:
        lines.append(f"\nModules ({len(modules)}):")
        for m in modules:
            cap = f" (capacity: {m.get('capacity')})" if m.get("capacity") else ""
            lines.append(f"  • {m.get('name', m.get('symbol', '?'))}{cap}")
    else:
        lines.append("\nModules: None")

    # Capabilities summary
    caps = _get_ship_capabilities(ship)
    if caps:
        lines.append(f"\nCapabilities: {', '.join(caps)}")

    return "\n".join(lines)


@tool
def scan_system(
    system_symbol: str,
    reference_ship: str | None = None,
    closest_only: bool = False,
    within_cruise_range: bool = False,
) -> str:
    """Scan all waypoints in a system to discover markets, shipyards, and resources WITHOUT requiring ship visits.

    This is VERY efficient - one API call reveals structural market data (what each market imports/exports) for the entire system.
    Use this early to understand the system's economy before moving ships around.

    Args:
        system_symbol: System to scan (e.g., 'X1-AB12') or waypoint (e.g., 'X1-AB12-C3') - will extract system automatically
        reference_ship: Ship symbol to calculate distances from (e.g., 'WHATER-1')
                       If provided, results are ALWAYS sorted by distance (closest first)
        closest_only: If True, return only the closest waypoint of each category (1 market, 1 shipyard, 1 asteroid)
        within_cruise_range: If True and reference_ship provided, filter to only waypoints within ship's CRUISE fuel range

    The system_symbol looks like 'X1-AB12'. If you pass a waypoint like 'X1-AB12-C3', the system will be extracted.
    """

    # Extract system from waypoint if needed (e.g., 'X1-KD26-A1' -> 'X1-KD26')
    # System format: SECTOR-SYSTEM (e.g., X1-KD26)
    # Waypoint format: SECTOR-SYSTEM-WAYPOINT (e.g., X1-KD26-A1)
    parts = system_symbol.split("-")
    if len(parts) > 2:
        # This is a waypoint, extract system (first two parts)
        system_symbol = f"{parts[0]}-{parts[1]}"

    # Get all waypoints in the system
    waypoints = get_system_waypoints(system_symbol)
    if isinstance(waypoints, dict) and "error" in waypoints:
        return f"Error: {waypoints['error']}"
    if not waypoints:
        return f"No waypoints found in system {system_symbol}."

    # Get reference ship position and fuel if specified
    reference_position = None
    reference_ship_name = None
    ship_fuel_capacity = 0

    if reference_ship:
        try:
            ship_status = _get_local_ship(reference_ship)
            current_wp = ship_status.location or ""
            ship_fuel_capacity = ship_status.fuel_capacity

            # Find reference ship's coordinates
            for wp in waypoints:
                if wp.get("symbol") == current_wp:
                    reference_position = (wp.get("x", 0), wp.get("y", 0))
                    reference_ship_name = reference_ship
                    break
        except Exception:
            pass

    # Helper function to calculate distance
    def calc_distance(wp):
        if not reference_position:
            return 0
        wx, wy = wp.get("x", 0), wp.get("y", 0)
        rx, ry = reference_position
        return calculate_distance(wx, wy, rx, ry)

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
                            entry[key] = [
                                i.get("symbol") if isinstance(i, dict) else str(i)
                                for i in items
                            ]

                # Save to cache if we found any market data
                if entry:
                    cache[wp_symbol] = entry
                    MARKET_CACHE_FILE.write_text(
                        json.dumps(cache, indent=2), encoding="utf-8"
                    )

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
        market_waypoints = [
            wp for wp in market_waypoints if calc_distance(wp) <= max_distance
        ]
        shipyard_waypoints = [
            wp for wp in shipyard_waypoints if calc_distance(wp) <= max_distance
        ]
        asteroid_waypoints = [
            wp for wp in asteroid_waypoints if calc_distance(wp) <= max_distance
        ]
        lines.append(f"Filtered to within CRUISE range ({max_distance} units):\n")

    # Limit to closest only if requested
    if closest_only:
        market_waypoints = market_waypoints[:1]
        shipyard_waypoints = shipyard_waypoints[:1]
        asteroid_waypoints = asteroid_waypoints[:1]

    # Summary
    if reference_ship_name:
        lines.append(f"System Scan (distances from {reference_ship_name}):")
    lines.append(
        f"Markets: {len(market_waypoints)} | Shipyards: {len(shipyard_waypoints)} | Asteroids: {len(asteroid_waypoints)}\n"
    )

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
                imports = [
                    i.get("symbol") if isinstance(i, dict) else str(i)
                    for i in wp.get("imports", [])
                ]
                if imports:
                    lines.append(f"  Imports (buys): {', '.join(imports[:10])}")
            if "exports" in wp:
                exports = [
                    e.get("symbol") if isinstance(e, dict) else str(e)
                    for e in wp.get("exports", [])
                ]
                if exports:
                    lines.append(f"  Exports (sells): {', '.join(exports[:10])}")
            if "exchange" in wp:
                exchange = [
                    e.get("symbol") if isinstance(e, dict) else str(e)
                    for e in wp.get("exchange", [])
                ]
                if exchange:
                    lines.append(f"  Exchange: {', '.join(exchange)}")

    # Show shipyards
    if shipyard_waypoints:
        lines.append("\n=== SHIPYARDS ===")
        display_shipyards = (
            shipyard_waypoints[:5] if not closest_only else shipyard_waypoints
        )
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
        display_asteroids = (
            asteroid_waypoints[:5] if not closest_only else asteroid_waypoints
        )

        for wp in display_asteroids:
            dist_str = ""
            if reference_position:
                dist = calc_distance(wp)
                dist_str = f" (distance from {reference_ship_name}: {dist:.1f})"

            traits = [
                t.get("symbol", "")
                for t in wp.get("traits", [])
                if t.get("symbol") not in ["MARKETPLACE", "SHIPYARD"]
            ]
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
    system_symbol = "-".join(waypoint_symbol.split("-")[:2])

    data = client.get_shipyard(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    ships = data.get("ships", data.get("shipTypes", []))
    # Save to cache immediately so purchase_ship knows about it
    _save_shipyard_to_cache(waypoint_symbol, data)

    lines = [f"Shipyard at {waypoint_symbol}:"]
    if isinstance(ships, list) and ships:
        for s in ships:
            if isinstance(s, dict) and "name" in s:
                lines.append(
                    f"  {s.get('type', '?')} ({s.get('name', '?')}) — {s.get('purchasePrice', '?')} credits"
                )
            else:
                lines.append(f"  {s.get('type', str(s))}")
    else:
        lines.append(
            "  No ship details available (need a ship present at this waypoint to see prices)."
        )
    return "\n".join(lines)


@tool
def view_market(waypoint_symbol: str) -> str:
    """View market prices and shipyard info at a waypoint. Always returns complete info:
    live prices if a ship is present, otherwise cached prices with staleness indicator.

    Args:
        waypoint_symbol: Waypoint with market (e.g., 'X1-KD26-B7'), or "ALL" to view all cached markets
    """
    import time

    # Handle "ALL" special case — return all cached markets without API calls
    if waypoint_symbol.upper() == "ALL":
        cache = load_market_cache()
        lines = ["=== All Known Markets ==="]

        # Collect all market lags for average calculation
        lags = []
        current_time = int(time.time())

        # Sort by system for better organization
        waypoints_by_system = {}
        for wp, data in cache.items():
            if wp == "_systems_fetched":
                continue
            # Only include waypoints with MARKETPLACE trait
            if not data.get("has_market"):
                continue
            sys = "-".join(wp.split("-")[:2])
            waypoints_by_system.setdefault(sys, []).append((wp, data))

        # Display markets grouped by system
        for sys_sym in sorted(waypoints_by_system.keys()):
            lines.append(f"\n[{sys_sym}]")
            for wp, market_data in waypoints_by_system[sys_sym]:
                last_updated = market_data.get("last_updated")
                if last_updated:
                    age_sec = current_time - last_updated
                    lags.append(age_sec)
                    if age_sec < 3600:
                        age_str = f"{age_sec // 60}m"
                    elif age_sec < 86400:
                        age_str = f"{age_sec // 3600}h"
                    else:
                        age_str = f"{age_sec // 86400}d"
                else:
                    age_str = "?"

                goods_count = len(market_data.get("trade_goods", []))
                lines.append(f"  {wp}: {goods_count} goods ({age_str} old)")

        # Add average lag summary
        if lags:
            avg_lag_sec = sum(lags) // len(lags)
            if avg_lag_sec < 3600:
                avg_lag_str = f"{avg_lag_sec // 60}m"
            elif avg_lag_sec < 86400:
                avg_lag_str = f"{avg_lag_sec // 3600}h"
            else:
                avg_lag_str = f"{avg_lag_sec // 86400}d"
            lines.append(f"\nAverage market age: {avg_lag_str}")

        return "\n".join(lines)

    system_symbol = "-".join(waypoint_symbol.split("-")[:2])
    lines = [f"Market at {waypoint_symbol}:"]

    # Try live API data (succeeds with full prices only when a ship is present)
    api_data = None
    api_error = None
    try:
        result = client.get_market(system_symbol, waypoint_symbol)
        if isinstance(result, dict) and "error" in result:
            api_error = result["error"]
        elif result:
            api_data = result
            _save_market_to_cache(waypoint_symbol, api_data)
    except Exception as e:
        api_error = str(e)

    live_trade_goods = api_data.get("tradeGoods", []) if api_data else []

    # Load cache for structural info and fallback prices
    cache = load_market_cache()
    cached = cache.get(waypoint_symbol, {})

    # Imports / Exports / Exchange — prefer live, fall back to cache
    if api_data:
        for section, label in [
            ("exports", "Exports"),
            ("imports", "Imports"),
            ("exchange", "Exchange"),
        ]:
            items = api_data.get(section, [])
            if items:
                lines.append(
                    f"  {label}: {', '.join(i['symbol'] if isinstance(i, dict) else i for i in items)}"
                )
    elif cached:
        for section, label in [
            ("exports", "Exports"),
            ("imports", "Imports"),
            ("exchange", "Exchange"),
        ]:
            items = cached.get(section, [])
            if items:
                lines.append(f"  {label}: {', '.join(items)}")

    # Price data — live if available, else cached with age
    if live_trade_goods:
        lines.append("  Prices (live):")
        for g in live_trade_goods:
            lines.append(
                f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} (vol: {g.get('tradeVolume', '?')})"
            )
    else:
        cached_goods = cached.get("trade_goods", [])
        if cached_goods:
            last_updated = cached.get("last_updated")
            if last_updated:
                age_sec = int(time.time()) - last_updated
                if age_sec < 3600:
                    age_str = f"{age_sec // 60}m ago"
                elif age_sec < 86400:
                    age_str = f"{age_sec // 3600}h ago"
                else:
                    age_str = f"{age_sec // 86400}d ago"
            else:
                age_str = "age unknown"
            lines.append(f"  Prices (cached, {age_str}):")
            for g in cached_goods:
                lines.append(
                    f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} (vol: {g.get('tradeVolume', '?')})"
                )
        else:
            lines.append("  No price data (no ship present, nothing cached).")
            if api_error:
                lines.append(f"  API error: {api_error}")

    # Shipyard data (only if waypoint has a shipyard)
    if cached.get("has_shipyard"):
        try:
            shipyard = client.get_shipyard(system_symbol, waypoint_symbol)
            if isinstance(shipyard, dict) and "error" not in shipyard and shipyard:
                _save_shipyard_to_cache(waypoint_symbol, shipyard)
                ships = shipyard.get("ships", shipyard.get("shipTypes", []))
                lines.append(f"\nShipyard at {waypoint_symbol}:")
                if isinstance(ships, list) and ships:
                    for s in ships:
                        if isinstance(s, dict) and "name" in s:
                            lines.append(
                                f"  {s.get('type', '?')} ({s.get('name', '?')}) — {s.get('purchasePrice', '?')} credits"
                            )
                        else:
                            lines.append(f"  {s.get('type', str(s))}")
                else:
                    lines.append(
                        "  No ship details available (need a ship present to see prices)."
                    )
        except Exception:
            pass

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Action tools
# ──────────────────────────────────────────────


@tool
def accept_contract(contract_id: str) -> str:
    """[STATE: credits, contract status] Accept a contract. Gives upfront credit payment."""
    data = client.accept_contract(contract_id)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    _intercept_agent(data)
    agent = data.get("agent", {})
    contract = data.get("contract", {})
    return (
        f"Contract {contract.get('id', contract_id)} accepted!\n"
        f"Credits now: {agent.get('credits', '?')}"
    )


@tool
def orbit_ship(ship_symbol: str) -> str:
    """[STATE: nav_status] Put a ship into orbit. Required before navigating or extracting."""
    data = client.orbit(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    _intercept(ship_symbol, data)
    nav = data.get("nav", {})
    return f"{ship_symbol} is now in orbit at {nav.get('waypointSymbol', '?')}."


# ──────────────────────────────────────────────
#  Navigation & Action tools
# ──────────────────────────────────────────────


@tool
def navigate_ship(
    ship_symbol: str, destination_symbol: str, mode: str = "CRUISE"
) -> str:
    """
    [STATE: position, fuel] Smart Navigation.
    - If destination is close: Navigates directly.
    - If destination requires multi-hop: ENGAGES AUTOPILOT (assigns a temporary 'goto' behavior).
    The ship will automatically refuel and hop until it reaches the destination.
    """
    try:
        # 1. Check if this requires multi-hop
        ship_status = _get_local_ship(ship_symbol)
        current_wp = ship_status.location or ""
        current_sys = current_wp.rsplit("-", 1)[0] if current_wp else ""

        if current_wp == destination_symbol:
            return f"{ship_symbol} is already at {destination_symbol}."

        # Fetch route details without executing to check distance/hops
        # We assume standard CRUISE for calculation if mode not specified
        waypoints = get_system_waypoints(current_sys)
        origin_obj = next((w for w in waypoints if w["symbol"] == current_wp), None)
        target_obj = next(
            (w for w in waypoints if w["symbol"] == destination_symbol), None
        )

        if origin_obj and target_obj:
            ship_dict = {
                "fuel": {"current": ship_status.fuel_current, "capacity": ship_status.fuel_capacity},
                "engine": {"speed": ship_status.engine_speed},
            }
            path = _find_refuel_path(ship_dict, origin_obj, target_obj, waypoints, mode)

            # 2. If multi-hop route found (length > 2 means Origin -> Stop -> Dest), assign behavior
            if path and len(path) > 2:
                engine = get_engine()
                # Create a "One-Shot" behavior: Go to destination, then Stop (return to manual)
                # We use the behavior engine's robustness to handle the hops/refueling
                cmds = f"goto {destination_symbol} {mode}, stop"

                # Check if ship already has a behavior we shouldn't overwrite?
                # For now, we assume explicit tool call overrides existing behavior
                engine.assign(ship_symbol, cmds)

                stops = len(path) - 2
                return (
                    f"🚀 AUTOPILOT ENGAGED for {ship_symbol}. Multi-hop route detected ({stops} stops).\n"
                    f"Assigned behavior: '{cmds}'\n"
                    f"Ship will auto-refuel and travel to {destination_symbol}. check 'view_ships' for progress."
                )

        # 3. If direct route or simple jump, just execute standard logic
        msg, wait = _navigate_ship_logic(
            ship_symbol, destination_symbol, mode, execute=True
        )
        navigate_ship._last_wait = wait
        return msg
    except Exception as e:
        return f"Error: {e}"


# Initialize attribute
navigate_ship._last_wait = 0.0


@tool
def plan_route(ship_symbol: str, destination: str, mode: str = "CRUISE") -> str:
    """
    [READ-ONLY] Calculate distance and fuel cost for a route.
    Shows comparison table for CRUISE/DRIFT/BURN.
    """
    try:
        result, _ = _navigate_ship_logic(ship_symbol, destination, mode, execute=False)
        return result
    except Exception as e:
        return f"Error: {e}"


@tool
def dock_ship(ship_symbol: str) -> str:
    """[STATE: nav_status] Dock a ship. Required before refueling, selling, or purchasing."""
    data = client.dock(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    _intercept(ship_symbol, data)
    nav = data.get("nav", {})
    return f"{ship_symbol} is now docked at {nav.get('waypointSymbol', '?')}."


@tool
def refuel_ship(ship_symbol: str) -> str:
    """[STATE: fuel, credits] Refuel a ship at current waypoint. Auto-docks. Waypoint must sell fuel."""
    try:
        return _refuel_ship_logic(ship_symbol)
    except Exception as e:
        return f"Error: {e}"


@tool
def buy_ship(ship_type: str, waypoint_symbol: str) -> str:
    """[STATE: credits, fleet] Purchase a ship at a shipyard.
    Common types: SHIP_PROBE, SHIP_MINING_DRONE, SHIP_SIPHON_DRONE, SHIP_LIGHT_HAULER,
    SHIP_COMMAND_FRIGATE, SHIP_EXPLORER, SHIP_HEAVY_FREIGHTER, SHIP_ORE_HOUND, SHIP_REFINING_FREIGHTER.
    """
    try:
        result = _buy_ship_logic(ship_type, waypoint_symbol)
        ship = result["ship"]
        credits = result["credits_remaining"]
        return (
            f"Purchased {ship.get('symbol', '?')} ({ship_type})!\n"
            f"Credits remaining: {credits}"
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def extract_ore(ship_symbol: str) -> str:
    """[STATE: cargo, cooldown] Extract ores from an asteroid. Auto-orbits. Cooldown applies between extractions."""
    try:
        msg, wait = _extract_ore_logic(ship_symbol)
        extract_ore._last_wait = wait
        return msg
    except Exception as e:
        return f"Error: {e}"


# Initialize the wait tracking attribute
extract_ore._last_wait = 0.0


@tool
def sell_cargo(
    ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False
) -> str:
    """[STATE: cargo, credits] Sell cargo at current market. Auto-docks. Refuses to sell contract goods."""
    try:
        return _sell_cargo_logic(ship_symbol, trade_symbol, units, force)
    except Exception as e:
        return f"Error: {e}"


@tool
def buy_cargo(ship_symbol: str, trade_symbol: str, units: int = None) -> str:
    """[STATE: cargo, credits] Buy cargo from current market. Auto-docks. Fills remaining cargo space by default."""
    try:
        return _buy_cargo_logic(ship_symbol, trade_symbol, units)
    except Exception as e:
        return f"Error: {e}"


@tool
def jettison_cargo(
    ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False
) -> str:
    """[STATE: cargo] Jettison cargo into space. Refuses to jettison contract goods unless force=True."""
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            return (
                f"Error: {trade_symbol} is required by an active contract. "
                f"Use deliver_contract to deliver it instead. "
                f"Call jettison_cargo with force=True to jettison anyway."
            )

    # Use local tracker to check inventory (0 API calls)
    try:
        ship = _get_local_ship(ship_symbol)
    except Exception as e:
        return f"Error: {e}"

    inventory = ship.cargo_inventory
    available = 0

    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        return f"Error: Ship {ship_symbol} has no {trade_symbol} available."

    safe_units = available if units is None else min(units, available)

    data = client.jettison(ship_symbol, trade_symbol, safe_units)
    _intercept(ship_symbol, data)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    cargo = data.get("cargo", {})
    return (
        f"Jettisoned {safe_units} {trade_symbol}.\n"
        f"Cargo now: {cargo.get('units', 0)}/{cargo.get('capacity', '?')} units"
    )


@tool
def transfer_cargo(
    from_ship: str, to_ship: str, trade_symbol: str, units: int = None
) -> str:
    """[STATE: cargo] Transfer cargo between ships. Auto-orbits both. Both must be at same waypoint."""
    try:
        return _transfer_cargo_logic(from_ship, to_ship, trade_symbol, units)
    except Exception as e:
        return f"Error: {e}"


# ──────────────────────────────────────────────
#  Advanced ship operations
# ──────────────────────────────────────────────


@tool
def survey_asteroid(ship_symbol: str) -> str:
    """[STATE: cooldown] Survey an asteroid field for rich deposits. Ship must be in orbit at an asteroid."""
    data = client.survey(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    surveys = data.get("surveys", [])
    cooldown = data.get("cooldown", {})
    lines = [
        f"Created {len(surveys)} survey(s). Cooldown: {cooldown.get('remainingSeconds', 0)}s"
    ]
    for i, survey in enumerate(surveys):
        deposits = [d.get("symbol", "?") for d in survey.get("deposits", [])]
        lines.append(
            f"  Survey {i+1}: {survey.get('signature', '?')} - Size: {survey.get('size', '?')}"
        )
        lines.append(f"    Deposits: {', '.join(deposits)}")
        lines.append(f"    Expires: {survey.get('expiration', '?')}")
    return "\n".join(lines)


@tool
def scan_waypoints(ship_symbol: str) -> str:
    """[STATE: cooldown] Scan for waypoints from current location. Ship must be in orbit."""
    data = client.scan_waypoints(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    waypoints = data.get("waypoints", [])
    cooldown = data.get("cooldown", {})
    lines = [
        f"Found {len(waypoints)} waypoints. Cooldown: {cooldown.get('remainingSeconds', 0)}s"
    ]
    for wp in waypoints:
        traits = ", ".join(t.get("symbol", "") for t in wp.get("traits", [])[:3])
        lines.append(f"  {wp.get('symbol', '?')} ({wp.get('type', '?')}) - {traits}")
    return "\n".join(lines)


@tool
def scan_ships(ship_symbol: str) -> str:
    """[STATE: cooldown] Scan for other ships in the area. Ship must be in orbit."""
    data = client.scan_ships(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    ships = data.get("ships", [])
    cooldown = data.get("cooldown", {})
    lines = [
        f"Found {len(ships)} ships. Cooldown: {cooldown.get('remainingSeconds', 0)}s"
    ]
    for ship in ships[:10]:
        nav = ship.get("nav", {})
        lines.append(
            f"  {ship.get('symbol', '?')} - {ship.get('registration', {}).get('role', '?')} @ {nav.get('waypointSymbol', '?')}"
        )
    return "\n".join(lines)


@tool
def jump_ship(ship_symbol: str, waypoint_symbol: str) -> str:
    """[STATE: position, cooldown] Jump to another star system via jump gate. Requires antimatter."""
    _ensure_orbit_logic(ship_symbol)
    data = client.jump(ship_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    cooldown = data.get("cooldown", {})
    return (
        f"{ship_symbol} jumped to system {waypoint_symbol}.\n"
        f"Now at: {nav.get('waypointSymbol', '?')}\n"
        f"Cooldown: {cooldown.get('remainingSeconds', 0)}s"
    )


@tool
def warp_ship(ship_symbol: str, waypoint_symbol: str) -> str:
    """[STATE: position, fuel] Warp to a waypoint in another system. Uses antimatter."""
    _ensure_orbit_logic(ship_symbol)
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
    """[STATE: contracts] Negotiate a new contract. Creates a behavior to go to HQ and negotiate."""
    # Get HQ location from agent
    agent = _get_local_agent()
    hq_waypoint = agent.get("headquarters")
    if not hq_waypoint:
        return "Error: Could not find headquarters waypoint"

    # Create behavior: goto HQ, negotiate, stop
    steps_str = f"goto {hq_waypoint}, negotiate, stop"
    result = get_engine().assign(ship_symbol, steps_str)

    return f"Assigned negotiation behavior to {ship_symbol}: {steps_str}\n{result}"


@tool
def chart_waypoint(ship_symbol: str) -> str:
    """[STATE: waypoint data] Chart the current waypoint to add it to known waypoints."""
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
    system_symbol = "-".join(waypoint_symbol.split("-")[:2])

    data = client.get_jump_gate(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    connections = data.get("connections", [])
    lines = [f"Jump gate at {waypoint_symbol} connects to {len(connections)} systems:"]
    for conn in connections:
        lines.append(f"  → {conn}")
    return "\n".join(lines)


@tool
def view_construction(waypoint_symbol: str) -> str:
    """[READ-ONLY] View construction project details at a waypoint. Shows progress, materials needed, and requirements.

    Args:
        waypoint_symbol: Waypoint with construction (e.g., 'X1-ABC-123A')
    """
    # Extract system from waypoint (e.g., 'X1-ABC-123A' -> 'X1-ABC')
    system_symbol = "-".join(waypoint_symbol.split("-")[:2])

    data = client.get_construction(system_symbol, waypoint_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    lines = [f"Construction at {waypoint_symbol}:"]

    # Project symbol
    symbol = data.get("symbol", "?")
    lines.append(f"  Symbol: {symbol}")

    # Progress
    progress = data.get("progress", 0)
    required = data.get("required", 0)
    if required > 0:
        pct = (progress / required * 100) if required > 0 else 0
        lines.append(f"  Progress: {progress}/{required} ({pct:.1f}%)")
    else:
        lines.append(f"  Progress: {progress}")

    # Materials
    materials = data.get("materials", [])
    if materials:
        lines.append("  Materials needed:")
        for mat in materials:
            trade_symbol = mat.get("tradeSymbol", "?")
            required_amount = mat.get("required", 0)
            fulfilled_amount = mat.get("fulfilled", 0)
            pct = (
                (fulfilled_amount / required_amount * 100) if required_amount > 0 else 0
            )
            lines.append(
                f"    {trade_symbol}: {fulfilled_amount}/{required_amount} ({pct:.1f}%)"
            )

    # Deadline
    deadline = data.get("deadline", None)
    if deadline:
        lines.append(f"  Deadline: {deadline}")

    return "\n".join(lines)


@tool
def deliver_contract(
    contract_id: str, ship_symbol: str, trade_symbol: str, units: int = None
) -> str:
    """[STATE: cargo, contract] Deliver goods for a contract. Auto-docks. Intelligently calculates optimal delivery amount.

    If units is specified, delivers that amount (capped by contract requirements and available cargo).
    If units is omitted, delivers as much as possible (limited by contract requirements and cargo available).

    Example:
      deliver_contract("cmm6x17d5jc5dui6zvdoudt2o", "WHATER-1", "DIAMONDS")  # Deliver all available DIAMONDS (up to contract limit)
      deliver_contract("cmm6x17d5jc5dui6zvdoudt2o", "WHATER-1", "DIAMONDS", 5)  # Deliver 5 (or less if not enough)
    """
    try:
        return _deliver_contract_logic(contract_id, ship_symbol, trade_symbol, units)
    except Exception as e:
        return f"Error: {e}"


@tool
def fulfill_contract(contract_id: str) -> str:
    """[STATE: credits, contract] Complete a contract after all deliveries. Collects final payment."""
    data = client.fulfill_contract(contract_id)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"

    _intercept_agent(data)
    agent = data.get("agent", {})
    return (
        f"Contract {contract_id} fulfilled!\n"
        f"Credits now: {agent.get('credits', '?')}"
    )


@tool
def find_waypoints(
    waypoint_type: str | None = None,
    trait: str | None = None,
    trade_symbol: str | None = None,
    system_symbol: str | None = None,
    reference_ship: str | None = None,
) -> str:
    """Find locations of interest. Merges functionality of finding waypoints, markets, and nearest resources.

    Args:
        waypoint_type: Filter by type (e.g., ASTEROID, PLANET, GAS_GIANT).
        trait: Filter by trait (e.g., MARKETPLACE, SHIPYARD, COMMON_METAL_DEPOSITS).
        trade_symbol: Find markets buying/selling this good (e.g., FUEL, IRON_ORE). Uses cached data.
        system_symbol: System to search. Defaults to reference_ship's system or agent's HQ system.
        reference_ship: Sort results by distance from this ship.
    """
    # 1. Determine System
    ref_coords = None
    if reference_ship:
        try:
            ship_status = _get_local_ship(reference_ship)
            ship_wp = ship_status.location or ""
            if ship_wp:
                system_symbol = ship_wp.rsplit("-", 1)[0]

                # Fetch waypoints to get coords of current location
                cache = load_waypoint_cache()
                if ship_wp in cache:
                    ref_coords = (cache[ship_wp].get("x", 0), cache[ship_wp].get("y", 0))
        except Exception:
            pass

    if not system_symbol:
        # Fallback to agent HQ system
        agent = _get_local_agent()
        if "headquarters" in agent:
            system_symbol = "-".join(agent["headquarters"].split("-")[:2])
        else:
            return "Error: system_symbol or reference_ship must be provided."

    # 2. Call Logic
    results = _find_waypoints_logic(
        system_symbol=system_symbol,
        waypoint_type=waypoint_type,
        trait=trait,
        trade_symbol=trade_symbol,
        ref_coords=ref_coords,
    )

    if not results:
        if trade_symbol:
            return f"No known markets for {trade_symbol} in {system_symbol}. (Note: Only checks cached data)."
        return f"No waypoints found in {system_symbol} matching your criteria."

    # 3. Format Output
    lines = []
    if trade_symbol:
        lines.append(f"Markets for {trade_symbol} in {system_symbol}:")
        for res in results:
            dist = res["distance"]
            sym = res["symbol"]
            mm = res["market_match"]

            d_str = f" ({dist:.1f} dist)" if dist != float("inf") else ""

            details = []
            if mm["purchasePrice"]:
                details.append(f"BUY: {mm['purchasePrice']}")
            if mm["sellPrice"]:
                details.append(f"SELL: {mm['sellPrice']}")
            if not details:  # Fallback to roles if no price
                details = mm["roles"]

            lines.append(f"  {sym}{d_str}: {', '.join(details)}")

    else:
        lines.append(f"Waypoints in {system_symbol}:")
        for res in results:
            wp = res["waypoint"]
            dist = res["distance"]
            d_str = f" ({dist:.1f} dist)" if dist != float("inf") else ""
            t_list = [t["symbol"] for t in wp.get("traits", [])]
            lines.append(
                f"  {wp['symbol']} [{wp['type']}]{d_str} - {', '.join(t_list)}"
            )

        # if len(results) > 20:
        #    lines.append(f"  ... {len(results)-20} more ...")

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Trade analysis
# ──────────────────────────────────────────────


@tool
def find_trades(ship_symbol: str | None = None, good: str | None = None, min_profit: int = 1) -> str:
    """[READ-ONLY] Find profitable trade routes from cached market data.
    Compares buy prices (exports/exchange) with sell prices (imports/exchange) across
    all known markets. Only uses cached price data — send ships to markets for fresh prices.
    Args:
        ship_symbol: Optional. If given, shows distance from this ship and sorts by profit/distance.
        good: Optional. Filter to a specific trade good (e.g. "IRON_ORE").
        min_profit: Minimum profit per unit to include (default: 1).
    """
    routes = _analyze_trade_routes(ship_symbol, min_profit)

    if not routes:
        return f"No profitable routes found (min_profit={min_profit}). Try lowering min_profit or scouting more markets."

    # Filter by good if requested
    if good:
        routes = [r for r in routes if r["good"] == good.upper()]
        if not routes:
            return f"No routes found for {good}."

    # Format top 10
    lines = ["Top trade routes (from cached prices):\n"]
    for r in routes[:10]:
        lines.append(
            f"  {r['good']}: buy at {r['src']} ({r['buy']}/unit) -> sell at {r['snk']} ({r['sell']}/unit)"
        )
        detail = f"    Profit: {r['profit']}/unit | Volume: {r['volume']}"
        if r.get("dist"):
            detail += f" | Dist: {r['dist']:.1f} from {ship_symbol}"
        if r["stale"]:
            detail += " | STALE PRICES (2h+)"
        lines.append(detail)
    return "\n".join(lines)


# ──────────────────────────────────────────────
#  Planning tool
# ──────────────────────────────────────────────


@tool
def update_plan(plan: str) -> str:
    """[STATE: plan file] Write or update your plan. Shown in [Current Plan] every turn. Visible to the operator."""
    import time
    from pathlib import Path

    plan_file = Path("plan.txt")

    # Check if plan was just updated (within last 60 seconds)
    if plan_file.exists():
        mtime = plan_file.stat().st_mtime
        age_seconds = time.time() - mtime
        if age_seconds < 60:
            return (
                f"ERROR: Plan was updated {int(age_seconds)}s ago! "
                f"STOP PLANNING and START EXECUTING the existing plan. "
                f"Read [Current Plan] and take the next action."
            )

    plan_file.write_text(plan, encoding="utf-8")
    return f"Plan updated ({len(plan)} chars). NOW EXECUTE IT - do not plan again!"


# ──────────────────────────────────────────────
#  Behavior tools (step-sequence engine)
# ──────────────────────────────────────────────


@tool
def create_behavior(ship_symbol: str, steps: str, start_step: int = 0) -> str:
    """[STATE: behavior] Create an automated step-sequence behavior for a ship.

    Steps execute in order automatically. Each step auto-handles dock/orbit.

    Steps (comma-separated):
      mine WAYPOINT [ORE1 ORE2]  - Navigate to asteroid, mine until cargo full, jettison non-targets
      goto WAYPOINT              - Navigate to waypoint, wait for arrival
      buy ITEM [UNITS] [max:PRICE] [min_qty:N] - Buy cargo from current market (fills remaining space by default).

                               max:PRICE stops and alerts if price per unit exceeds limit.
                               min_qty:N alerts if fewer than N units could be purchased.
      sell ITEM [min:PRICE]       - Sell cargo at current market (skips contract goods). min:PRICE sets a price floor — stops selling if price drops below
      deliver CONTRACT ITEM [N]  - Deliver cargo for contract (smart: auto-caps at contract remaining + cargo available)
      refuel                     - Refuel at current market
      scout                      - View market prices at current location
      chart                      - Chart the current location
      buy_ship SHIP_TYPE         - Buy a ship at current shipyard (e.g., buy_ship SHIP_LIGHT_HAULER)
      alert MESSAGE              - Pause and alert you
      repeat                     - Restart from step 1
      stop                       - End behavior, return ship to IDLE (manual control)

    Args:
        start_step: Step index to begin at (0-based). Use to spread multiple ships
                    across the same sequence so they don't converge on the first step.

    Example: "goto X1-MKT, buy IRON_ORE, goto X1-HUB, sell IRON_ORE, stop"
    """
    result = get_engine().assign(ship_symbol, steps, start_step=start_step)
    return result


@tool
def pause_behavior(ship_symbol: str) -> str:
    """[STATE: behavior] Pause a ship's behavior without advancing. Resume with resume_behavior when ready."""
    return get_engine().pause(ship_symbol)


@tool
def resume_behavior(ship_symbol: str) -> str:
    """[STATE: behavior] Resume a paused behavior after handling an alert. Does not advance the step."""
    return get_engine().resume(ship_symbol)


@tool
def skip_step(ship_symbol: str) -> str:
    """[STATE: behavior] Skip the current step of a behavior and advance to the next one."""
    return get_engine().skip_step(ship_symbol)


@tool
def cancel_behavior(ship_symbol: str) -> str:
    """[STATE: behavior] Cancel a ship's behavior. Ship returns to IDLE (manual LLM control).

    Cancel before manually operating a ship that has an active behavior.
    """
    engine = get_engine()
    if ship_symbol not in engine.behaviors:
        return f"{ship_symbol} has no assigned behavior."
    engine.cancel(ship_symbol)
    return f"Cancelled behavior for {ship_symbol}. Ship is now idle (manual control)."


@tool
def assign_mining_loop(
    ship_symbol: str, asteroid_wp: str, ore_types: str = "", start_step: int = 0
) -> str:
    """[STATE: behavior] Convenience: assign a mine-sell loop. Builds a step sequence internally.

    Args:
        ship_symbol: Ship to assign.
        asteroid_wp: Asteroid waypoint to mine at.
        ore_types: Comma-separated ore symbols to KEEP (e.g. "IRON_ORE,COPPER_ORE").
    """
    ore_list = (
        [s.strip() for s in ore_types.split(",") if s.strip()] if ore_types else []
    )
    ore_str = " ".join(ore_list) if ore_list else ""
    mine_part = f"mine {asteroid_wp} {ore_str}".strip()
    # Simple mine loop: mine until full, then alert for LLM to handle selling
    steps_str = f"{mine_part}, alert cargo full, repeat"
    return get_engine().assign(ship_symbol, steps_str, start_step)


@tool
def assign_trade_route(
    ship_symbol: str,
    buy_waypoint: str,
    buy_good: str,
    sell_waypoint: str,
    sell_good: str | None = None,
    end_step: str = "stop",
    start_step: int = 0,
) -> str:
    """[STATE: behavior] Assign a buying and selling route.
    The ship will:
    1. Go to buy_waypoint
    2. Buy the specified good (attempt to fill cargo) (automatically sets max price)
    3. Go to sell_waypoint
    4. Sell the good (automatically sets min price)
    5. stop (or repeat if end_step="repeat")
    Smart navigation handles refueling automatically.
    Args:
        ship_symbol: The ship.
        buy_waypoint: Where to buy (e.g., 'X1-KD26-D44').
        buy_good: The symbol to trade (e.g., 'SHIP_PARTS').
        sell_waypoint: Where to sell.
        sell_good: Optional. Defaults to buy_good. Use if refining/transforming, otherwise leave empty.
        end_step: Optional. Either "stop" or "repeat", defaults to "stop"
    """
    s_good = sell_good if sell_good else buy_good
    cache = load_market_cache()

    # Buy step: alert if price spikes beyond expected margin
    buy_step = f"buy {buy_good}"
    buy_market = cache.get(buy_waypoint, {})
    for good in buy_market.get("trade_goods", []):
        if good.get("symbol") == buy_good:
            buy_cost = good.get("purchasePrice")
            if buy_cost:
                max_buy = int(buy_cost * (1.0 + TRADE_PROFIT_MARGIN))
                buy_step = f"buy {buy_good} max:{max_buy}"
            break

    # Sell step: alert if price drops below expected margin
    sell_step = f"sell {s_good}"
    sell_market = cache.get(sell_waypoint, {})
    for good in sell_market.get("trade_goods", []):
        if good.get("symbol") == s_good:
            sell_price = good.get("sellPrice")
            if sell_price:
                min_sell = int(sell_price * (1.0 - TRADE_PROFIT_MARGIN))
                sell_step = f"sell {s_good} min:{min_sell}"
            break

    steps_str = f"goto {buy_waypoint}, {buy_step}, goto {sell_waypoint}, {sell_step}, {end_step}"
    return get_engine().assign(ship_symbol, steps_str, start_step)


@tool
def assign_satellite_scout(ship_symbols: str, market_waypoints: str = "") -> str:
    """[STATE: behavior] Assign smart scouting to satellites.
    Args:
        ship_symbols: Comma-separated satellite ship symbols.
        market_waypoints: Optional comma-separated waypoints. If omitted, ships will dynamically pick the best target based on distance and cache staleness.
    """
    ships = [s.strip() for s in ship_symbols.split(",") if s.strip()]
    if not ships:
        return "Error: no ship symbols provided."

    engine = get_engine()
    results = []

    for ship_sym in ships:
        if market_waypoints:
            # If a specific list is provided, just visit them and stop.
            markets = [m.strip() for m in market_waypoints.split(",") if m.strip()]
            parts = []
            for mkt in markets:
                parts.append(f"goto {mkt}")
                parts.append("scout")
            parts.append("stop")
            plan = ", ".join(parts)
        else:
            # Use the new JIT Probe Plan (phase-aware)
            try:
                ship_status = _get_local_ship(ship_sym)
                loc = ship_status.location or ""
                if loc:
                    strat = evaluate_fleet_strategy()
                    plan = _get_probe_plan(ship_sym, loc, strat["phase"])
                else:
                    plan = "explore"
            except Exception:
                plan = "explore"

        result = engine.assign(ship_sym, plan)
        results.append(result)

    return "\n".join(results)


@tool
def assign_auto_trade(ship_symbol: str) -> str:
    """[STATE: behavior] Assign the ship to automatically find and execute profitable trades.
    The ship will:
    1. Analyze cached market data for the best trade relative to its location.
    2. Fly to buy, buy goods, fly to sell, sell goods.
    3. Repeat step 1 forever.
    Requires cached market data (use satellites/scouting first).
    """
    return get_engine().assign(ship_symbol, "autotrade")


@tool
def assign_contract_duty(ship_symbol: str) -> str:
    """[STATE: behavior] Assign ship to contract duty: Goto HQ, negotiate/accept contracts, and automatically fulfill them.
    Loop: Fetch contract -> Buy goods -> Deliver -> Repeat.
    """
    # Simply assigning 'negotiate' triggers the smart logic:
    # If no contract -> goto HQ, negotiate. If contract exists -> fulfill it.
    return get_engine().assign(ship_symbol, "negotiate")


@tool
def assign_system_explorer(ship_symbol: str) -> str:
    """[STATE: behavior] Assign a probe/ship to automatically explore its current system.
    It will automatically fly to every waypoint, chart it, and scout all markets/shipyards.
    """
    return get_engine().assign(ship_symbol, "explore")


@tool
def assign_jump_gate_construction(ship_symbol: str) -> str:
    """[STATE: behavior] Assign a ship to automatically buy materials and construct the jump gate.
    Smart budgeting: will pause and alert if buying materials would dip into your required trade reserves!
    """
    return get_engine().assign(ship_symbol, "construct")


@tool
def toggle_hq(set: str | None = None, add: str | None = None, remove: str | None = None) -> str:
    """[STATE: hq] Configure which ships the HQ Fleet Director manages.

    Use 'set' to replace the entire target list, or 'add'/'remove' to modify the current list.

    Args:
        set:    Replace the target list entirely. Special values: "ALL" (all ships), "NONE" (disable HQ).
                Roles: HAULER, FREIGHTER, SATELLITE, COMMAND. Add BUY_SHIPS to allow purchasing.
                Examples: set="ALL", set="NONE", set="SATELLITE,HAULER,BUY_SHIPS"
        add:    Add tokens to the current list (comma-separated ship names or roles).
                Example: add="WHATER-1" or add="WHATER-1,BUY_SHIPS"
        remove: Remove tokens from the current list (comma-separated).
                Example: remove="HAULER" or remove="WHATER-2,BUY_SHIPS"

    Returns current HQ state after the change. You can call with no args to just read the current state.
    """
    current = _hq_managed_ships  # e.g. "SATELLITE,HAULER,BUY_SHIPS" or "ALL" or "NONE"

    if set is not None:
        new_targets = set.strip().upper()
    elif add is not None or remove is not None:
        # Build the current token set, treating ALL/NONE as opaque tokens
        if current in ("ALL", "NONE"):
            tokens = {current}
        else:
            tokens = {t.strip() for t in current.split(",") if t.strip()}

        if add:
            for t in add.upper().split(","):
                t = t.strip()
                if t:
                    tokens.add(t)
        if remove:
            for t in remove.upper().split(","):
                t = t.strip()
                tokens.discard(t)

        if not tokens:
            new_targets = "NONE"
        elif "ALL" in tokens:
            new_targets = "ALL"
        else:
            tokens.discard("NONE")
            new_targets = ",".join(sorted(tokens))
    else:
        new_targets = current  # no-op: just report status

    set_hq_enabled(new_targets)

    if new_targets == "NONE":
        status = "DISABLED (Manual control only)"
    elif new_targets == "ALL":
        status = "ENABLED for ALL ships"
    else:
        status = f"ENABLED for: {new_targets}"

    return f"HQ Fleet Director is now {status}."


# ──────────────────────────────────────────────
#  Tool registry
# ──────────────────────────────────────────────

# Tier 1: Essential tools
TIER_1_TOOLS = [
    # Navigation & planning
    navigate_ship,
    plan_route,
    # Trading & cargo
    buy_cargo,
    sell_cargo,
    transfer_cargo,
    jettison_cargo,
    # Contracts
    accept_contract,
    deliver_contract,
    fulfill_contract,
    negotiate_contract,
    # Info
    view_advisor,
    view_market,
    view_ships,
    view_contracts,
    find_trades,
    # Locator
    find_waypoints,
    # Planning
    update_plan,
    list_alerts,
    clear_alert,
    # Behavior control
    resume_behavior,
    skip_step,
    cancel_behavior,
    pause_behavior,
    assign_mining_loop,
    assign_satellite_scout,
    assign_trade_route,
    assign_auto_trade,
    assign_contract_duty,
    assign_system_explorer,
    assign_jump_gate_construction,
    create_behavior,
    toggle_hq,
]

# All tools (tier 2)
ALL_TOOLS = [
    # Observation
    view_agent,
    view_advisor,
    view_contracts,
    view_ships,
    view_ship_details,
    view_cargo,
    scan_system,
    view_market,
    view_jump_gate,
    view_construction,
    view_shipyards,
    # Locator
    find_waypoints,
    # Ship operations
    orbit_ship,
    dock_ship,
    navigate_ship,
    refuel_ship,
    buy_ship,
    plan_route,
    # Mining & resources
    extract_ore,
    survey_asteroid,
    # Trading & cargo
    buy_cargo,
    sell_cargo,
    jettison_cargo,
    transfer_cargo,
    # Contracts
    accept_contract,
    deliver_contract,
    fulfill_contract,
    negotiate_contract,
    # Scanning & exploration
    scan_waypoints,
    scan_ships,
    chart_waypoint,
    # Inter-system travel
    jump_ship,
    warp_ship,
    # Planning
    update_plan,
    # Analysis
    find_trades,
    list_alerts,
    clear_alert,
    # Behavior control
    create_behavior,
    resume_behavior,
    skip_step,
    cancel_behavior,
    pause_behavior,
    assign_mining_loop,
    assign_satellite_scout,
    assign_trade_route,
    assign_auto_trade,
    assign_contract_duty,
    assign_system_explorer,
    assign_jump_gate_construction,
    toggle_hq,
]

# Tools that are "significant" actions worth narrating
SIGNIFICANT_TOOLS = {
    "extract_ore",
    "survey_asteroid",
    "navigate_ship",
    "buy_cargo",
    "sell_cargo",
    "jettison_cargo",
    "transfer_cargo",
    "accept_contract",
    "fulfill_contract",
    "negotiate_contract",
    "refuel_ship",
    "buy_ship",
    "dock_ship",
    "orbit_ship",
    "deliver_contract",
    "scan_waypoints",
    "scan_ships",
    "chart_waypoint",
    "jump_ship",
    "warp_ship",
}

# Tools that have wait/cooldown times (for parallel narrative)
WAITING_TOOLS = {
    "navigate_ship",
    "extract_ore",
    "survey_asteroid",
    "scan_waypoints",
    "scan_ships",
    "jump_ship",
    "warp_ship",
}

# State-changing tools (tools that modify ship state — not read-only)
STATE_CHANGING_TOOLS = {
    "navigate_ship",
    "extract_ore",
    "sell_cargo",
    "jettison_cargo",
    "transfer_cargo",
    "refuel_ship",
    "buy_ship",
    "deliver_contract",
    "dock_ship",
    "orbit_ship",
    "survey_asteroid",
    "scan_waypoints",
    "scan_ships",
    "jump_ship",
    "warp_ship",
}


def get_tool_by_name(name: str):
    """Get a tool function by its name. Searches ALL_TOOLS (superset)."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
