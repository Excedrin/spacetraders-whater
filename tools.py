import json
import os
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv
from langchain_core.tools import tool
from api_client import SpaceTradersClient

# ──────────────────────────────────────────────
#  Global Config & Init
# ──────────────────────────────────────────────
load_dotenv()
client = SpaceTradersClient(os.environ["TOKEN"])
MARKET_CACHE_FILE = Path("market_cache.json")
BEHAVIORS_FILE = Path("behaviors.json")
log = logging.getLogger(__name__)

def load_market_cache() -> dict:
    if MARKET_CACHE_FILE.exists():
        try:
            return json.loads(MARKET_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _save_market_to_cache(waypoint_symbol: str, data: dict):
    """Save market data to cache."""
    import time
    cache = load_market_cache()
    entry = cache.get(waypoint_symbol, {})

    # Structural data (imports/exports/exchange)
    # Handle API variations where items might be dicts or strings
    for section in ("exports", "imports", "exchange"):
        items = data.get(section, [])
        if items:
            entry[section] = [i.get("symbol") if isinstance(i, dict) else str(i) for i in items]

    # Explicitly check for MARKETPLACE trait to ensure existence is cached
    # This ensures discover_all_markets populates the cache keys even if goods data is missing
    traits = data.get("traits", [])
    trait_symbols = [t.get("symbol") if isinstance(t, dict) else str(t) for t in traits]
    if "MARKETPLACE" in trait_symbols:
        entry["is_market"] = True  # Marker flag

    # Price data
    trade_goods = data.get("tradeGoods", [])
    if trade_goods:
        entry["trade_goods"] = [
            {
                "symbol": g["symbol"],
                "purchasePrice": g.get("purchasePrice"),
                "sellPrice": g.get("sellPrice"),
                "tradeVolume": g.get("tradeVolume")
            }
            for g in trade_goods
        ]
        entry["last_updated"] = int(time.time())

    # Save if we have any relevant data
    if entry:
        cache[waypoint_symbol] = entry
        MARKET_CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

# ──────────────────────────────────────────────
#  Core Logic Helpers (Shared)
# ──────────────────────────────────────────────
# These are pure logic functions used by both Tools and Behaviors.
# They return structured data (Exceptions or Tuples) rather than user-facing strings.

def _ensure_orbit_logic(ship_symbol: str) -> None:
    """Raises Exception if fails."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        raise Exception(f"Ship {ship_symbol} not found: {ship['error']}")

    status = ship.get("nav", {}).get("status", "")
    if status == "IN_ORBIT":
        return
    if status == "DOCKED":
        result = client.orbit(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            raise Exception(f"Could not orbit {ship_symbol}: {result['error']}")
        return
    if status == "IN_TRANSIT":
        raise Exception(f"{ship_symbol} is currently in transit")

class PriceFloorHit(Exception):
    """Raised when market price drops below minimum acceptable price."""
    def __init__(self, trade_symbol, current_price, min_price, sold, revenue):
        self.trade_symbol = trade_symbol
        self.current_price = current_price
        self.min_price = min_price
        self.sold = sold
        self.revenue = revenue
        super().__init__(f"{trade_symbol} price {current_price} dropped below floor {min_price}. Sold {sold} for {revenue} cr before stopping.")

class PriceCeilingHit(Exception):
    """Raised when market price exceeds maximum acceptable buy price."""
    def __init__(self, trade_symbol, current_price, max_price, bought, spent):
        self.trade_symbol = trade_symbol
        self.current_price = current_price
        self.max_price = max_price
        self.bought = bought
        self.spent = spent
        super().__init__(f"{trade_symbol} price {current_price} exceeded ceiling {max_price}. Bought {bought} for {spent} cr before stopping.")

class MinQtyNotMet(Exception):
    """Raised when buy step could not acquire the minimum required quantity."""
    def __init__(self, trade_symbol, bought, min_qty, reason=""):
        self.trade_symbol = trade_symbol
        self.bought = bought
        self.min_qty = min_qty
        self.reason = reason
        super().__init__(f"{trade_symbol}: only bought {bought} (need at least {min_qty}). {reason}".strip())

def _ensure_dock_logic(ship_symbol: str) -> None:
    """Raises Exception if fails."""
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        raise Exception(f"Ship {ship_symbol} not found: {ship['error']}")

    status = ship.get("nav", {}).get("status", "")
    if status == "DOCKED":
        return
    if status == "IN_ORBIT":
        result = client.dock(ship_symbol)
        if isinstance(result, dict) and "error" in result:
             raise Exception(f"Could not dock {ship_symbol}: {result['error']}")
        return
    if status == "IN_TRANSIT":
        raise Exception(f"{ship_symbol} is currently in transit")

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

def _calculate_travel_cost(ship: dict, dest_wp: dict, origin_wp: dict, mode: str = "CRUISE") -> tuple[int, int, int]:
    """
    Helper to calculate distance, fuel cost, and estimated time.
    Returns: (distance, fuel_cost, flight_seconds)
    """
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

def _find_refuel_path(ship: dict, origin_wp: dict, target_wp: dict, waypoints: list, mode="CRUISE") -> list[str] | None:
    """
    Perform BFS to find a path from origin to target using Marketplaces as refuel stops.
    Returns a list of waypoint symbols: [Origin, Stop1, Stop2, Target].
    """
    fuel_capacity = ship.get("fuel", {}).get("capacity", 0)
    current_fuel = ship.get("fuel", {}).get("current", 0)

    # 1. If solar (0 capacity), path is always direct.
    if fuel_capacity == 0:
        return [origin_wp['symbol'], target_wp['symbol']]

    # 2. Identify potential stops (Marketplaces)
    # Assumption: All marketplaces sell fuel.
    potential_stops = [w for w in waypoints if "MARKETPLACE" in [t['symbol'] for t in w.get('traits', [])]]

    # 3. BFS State: (current_wp_obj, path_list_of_symbols, fuel_at_current_node)
    queue = deque([(origin_wp, [origin_wp['symbol']], current_fuel)])
    visited = {origin_wp['symbol']}

    while queue:
        curr_node, path, fuel_available = queue.popleft()

        # A. Can we reach the Final Target from here?
        _, cost_to_target, _ = _calculate_travel_cost(ship, target_wp, curr_node, mode)
        if cost_to_target <= fuel_available:
            return path + [target_wp['symbol']]

        # B. If not, find reachable Marketplaces to hop to
        for stop in potential_stops:
            if stop['symbol'] in visited:
                continue

            _, cost_to_stop, _ = _calculate_travel_cost(ship, stop, curr_node, mode)

            if cost_to_stop <= fuel_available:
                visited.add(stop['symbol'])
                # We assume we refuel to FULL CAPACITY at the stop
                queue.append((stop, path + [stop['symbol']], fuel_capacity))

    return None

# ──────────────────────────────────────────────
#  CORE ACTION LOGIC (The "Smart" Layer)
# ──────────────────────────────────────────────

def _navigate_ship_logic(ship_symbol: str, destination_symbol: str, mode: str = "CRUISE", execute: bool = True) -> Tuple[str, float]:
    """
    Returns (result_message, wait_seconds).
    Handles smart routing, auto-refueling, and inter-system logic.
    """
    # 1. Structural Validation (Fast Fail)
    # A valid waypoint symbol MUST be SECTOR-SYSTEM-POINT (at least 2 hyphens).
    # This catches "WHATER-1" (ship) or "X1-RV42" (system only).
    if destination_symbol.count("-") < 2:
        raise Exception(f"Invalid destination format '{destination_symbol}'. Expected Waypoint Symbol (SECTOR-SYSTEM-WAYPOINT).")

    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        raise Exception(ship['error'])

    nav = ship.get("nav", {})
    fuel = ship.get("fuel", {})
    current_sys = nav.get("systemSymbol", "")
    current_wp = nav.get("waypointSymbol", "")

    if current_wp == destination_symbol:
        return f"{ship_symbol} is already at {destination_symbol}.", 0.0

    # Inter-System Check
    # Extract system from destination: X1-RV42-A1 -> X1-RV42
    dest_sys = "-".join(destination_symbol.split("-")[:2])
    target_wp_symbol = destination_symbol
    is_inter_system = False

    if current_sys != dest_sys:
        is_inter_system = True
        # FIX: Use type="JUMP_GATE" (not traits)
        waypoints = client.list_waypoints(current_sys, type="JUMP_GATE")

        if isinstance(waypoints, dict) and "error" in waypoints:
            raise Exception(f"API Error finding jump gate: {waypoints['error']}")

        if not waypoints:
             raise Exception(f"Destination is in {dest_sys}, but no Jump Gate found in {current_sys}.")

        target_wp_symbol = waypoints[0]['symbol']

        if current_wp == target_wp_symbol:
            return f"Ship is at Jump Gate. Use 'jump_ship' to jump to {dest_sys}.", 0.0

    # Fetch Waypoints to validate target exists and calculate stats
    waypoints = client.list_waypoints(current_sys)
    if isinstance(waypoints, dict) and "error" in waypoints:
        raise Exception(waypoints['error'])

    origin_obj = next((w for w in waypoints if w['symbol'] == current_wp), None)
    target_obj = next((w for w in waypoints if w['symbol'] == target_wp_symbol), None)

    # Local Existence Check
    if not origin_obj:
        raise Exception(f"Current location {current_wp} not found in system listing.")
    if not target_obj:
        # If we are staying in-system, this means the destination is bogus
        if not is_inter_system:
             raise Exception(f"Destination {target_wp_symbol} does not exist in system {current_sys}.")
        # If we are inter-system, target_obj is the Jump Gate, which must exist (checked above)
        raise Exception("Could not resolve Jump Gate coordinates.")

    # Execute Logic
    if execute:
        # 1. SMART REFUEL AT DEPARTURE
        # Only refuel if the current market sells fuel — avoids extra API calls
        # (and rate-limit pressure) when fuel is not available here.
        if fuel.get("capacity", 0) > 0:
            market_cache = load_market_cache()
            curr_market = market_cache.get(current_wp, {})
            if "FUEL" in curr_market.get("exchange", []) or "FUEL" in curr_market.get("exports", []):
                try:
                    _ensure_dock_logic(ship_symbol)
                    client.refuel(ship_symbol)
                    ship = client.get_ship(ship_symbol)
                    fuel = ship.get("fuel", {})
                except Exception:
                    pass

        _, direct_cost, direct_time = _calculate_travel_cost(ship, target_obj, origin_obj, mode)
        fuel_available = fuel.get("current", 0)
        fuel_capacity = fuel.get("capacity", 0)

        # 2. Route Check
        next_hop = target_wp_symbol
        is_multi_hop = False

        if fuel_capacity > 0 and direct_cost > fuel_available:
            path = _find_refuel_path(ship, origin_obj, target_obj, waypoints, mode)
            if not path:
                raise Exception(f"Stranded. Cannot reach {target_wp_symbol} ({direct_cost} fuel needed) and no refueling path found.")

            if len(path) > 1:
                next_hop = path[1]
                is_multi_hop = True
                next_hop_obj = next((w for w in waypoints if w['symbol'] == next_hop), None)
                _, _, direct_time = _calculate_travel_cost(ship, next_hop_obj, origin_obj, mode)

        # Action
        _ensure_orbit_logic(ship_symbol)
        if nav.get("flightMode") != mode:
            client.set_flight_mode(ship_symbol, mode)

        data = client.navigate(ship_symbol, next_hop)
        if isinstance(data, dict) and "error" in data:
            raise Exception(f"Error navigating: {data['error']}")

        wait_secs = _parse_arrival(data.get("nav", {}))
        # Navigate was called — transit MUST be > 0. If _parse_arrival returned 0
        # (e.g. arrival already past due to latency, or missing nav data),
        # fall back to the calculated estimate so transit always registers.
        if wait_secs <= 0:
            wait_secs = max(float(direct_time), 1.0)
        result = f"🚀 {ship_symbol} navigating to {next_hop} ({mode}). Est: {direct_time}s."

        if is_multi_hop:
            result += f"\nNote: Multi-hop route initiated. Stopping at {next_hop} to refuel."
        elif is_inter_system:
            result += f"\nNote: Arriving at Jump Gate. Use 'jump_ship' next."

        return result, wait_secs

    else:
        # 4. Planning Mode (Comparison Table)
        lines = [f"Route Plan: {current_wp} -> {target_wp_symbol}"]
        if is_inter_system:
             lines.append(f"Note: {target_wp_symbol} is the Jump Gate to reach {dest_sys}.")

        # Check Direct
        dist, direct_cost, direct_time = _calculate_travel_cost(ship, target_obj, origin_obj, "CRUISE")
        lines.append(f"Direct Distance: {dist}")

        fuel_cap = fuel.get("capacity", 0)
        fuel_curr = fuel.get("current", 0)

        lines.append("\nFlight Modes:")
        for m in ["CRUISE", "DRIFT", "BURN"]:
            path = _find_refuel_path(ship, origin_obj, target_obj, waypoints, mode=m)

            _, cost, time = _calculate_travel_cost(ship, target_obj, origin_obj, m)

            status = ""
            if fuel_cap > 0:
                if cost <= fuel_curr:
                    status = "✅ Direct"
                elif path:
                    stops = len(path) - 2 # Exclude start/end
                    status = f"✅ Multi-hop ({stops} stops: {'->'.join(path)})"
                else:
                    status = "❌ Impossible (Max range exceeded)"
            else:
                status = "✅ (Solar)"

            lines.append(f"  {m.ljust(7)}: {str(time).rjust(4)}s | Fuel: {str(cost).rjust(4)} | {status}")

        return "\n".join(lines), 0.0

def _extract_ore_logic(ship_symbol: str) -> Tuple[str, float]:
    """Returns (log_string, cooldown_seconds)."""
    # Check cooldown first
    cooldown = client.get_cooldown(ship_symbol)
    if isinstance(cooldown, dict) and not cooldown.get("error"):
        remaining = cooldown.get("remainingSeconds", 0)
        if remaining > 0:
             # We return this as a valid state, handled by caller
             return f"Cooldown remaining: {remaining}s", float(remaining)

    _ensure_orbit_logic(ship_symbol)
    data = client.extract(ship_symbol)
    print(data)

    if isinstance(data, dict) and "error" in data:
        err = data['error']
        err_msg = str(err)

        # Handle API error 4000 (Cooldown) specifically if API returns it as error
        # Some versions of the API/Client might structure this differently
        if "cooldown" in err_msg.lower() or (isinstance(err, dict) and err.get('code') == 4000):
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
    return result, float(cd.get("remainingSeconds", 0))

def _sell_cargo_logic(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False, min_price: int = None) -> str:
    # 1. Check Contract Safety
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            raise Exception(f"{trade_symbol} is required by an active contract. Use force=True to override.")

    # 2. Get ship location and cargo state
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        raise Exception(f"Error fetching ship: {ship['error']}")
    waypoint = ship.get("nav", {}).get("waypointSymbol")

    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo: {cargo_data['error']}")

    inventory = cargo_data.get("inventory", [])
    capacity = cargo_data.get("capacity", 0)
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        raise Exception(f"Ship {ship_symbol} has no {trade_symbol}.")

    target_units = available if units is None else min(units, available)

    # 3. Look up max transaction volume from market cache
    max_per_transaction = target_units
    if waypoint:
        market_cache = load_market_cache()
        if waypoint in market_cache:
            for good in market_cache[waypoint].get("trade_goods", []):
                if good.get("symbol") == trade_symbol:
                    vol = good.get("tradeVolume")
                    if vol and vol > 0:
                        max_per_transaction = vol
                    break

    # 4. Sell in chunks if needed
    _ensure_dock_logic(ship_symbol)
    total_sold = 0
    total_revenue = 0
    sell_count = 0

    while total_sold < target_units:
        units_to_sell = min(max_per_transaction, target_units - total_sold)

        data = client.sell_cargo(ship_symbol, trade_symbol, units_to_sell)
        if isinstance(data, dict) and "error" in data:
            if sell_count == 0:
                raise Exception(data['error'])
            break  # Partial success — stop and report what we sold

        tx = data.get("transaction", {})
        total_sold += tx.get("units", units_to_sell)
        total_revenue += tx.get("totalPrice", 0)
        sell_count += 1

        if min_price and sell_count > 0:
            price_per_unit = tx.get("pricePerUnit", 0)
            if price_per_unit < min_price:
                raise PriceFloorHit(trade_symbol, price_per_unit, min_price, total_sold, total_revenue)

    cargo = data.get("cargo", {})
    if sell_count > 1:
        return f"Sold {total_sold} {trade_symbol} for {total_revenue} cr ({sell_count} transactions). Cargo: {cargo.get('units')}/{cargo.get('capacity', capacity)}."
    else:
        return f"Sold {total_sold} {trade_symbol} for {total_revenue} cr. Cargo: {cargo.get('units')}/{cargo.get('capacity', capacity)}."

def _buy_cargo_logic(ship_symbol: str, trade_symbol: str, units: int = None, max_price: int = None, min_qty: int = None) -> str:
    """Buy cargo from the current market. Splits purchases across multiple transactions if needed.

    Respects cargo capacity, credits available, market transaction limits, and optional price/quantity guards.
    """
    # Get ship location and initial cargo state
    ship = client.get_ship(ship_symbol)
    if isinstance(ship, dict) and "error" in ship:
        raise Exception(f"Error fetching ship: {ship['error']}")

    waypoint = ship.get("nav", {}).get("waypointSymbol")
    if not waypoint:
        raise Exception(f"Cannot determine ship location")

    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo: {cargo_data['error']}")

    capacity = cargo_data.get("capacity", 0)
    current_units = cargo_data.get("units", 0)
    available_space = capacity - current_units

    if available_space <= 0:
        raise Exception(f"Ship {ship_symbol} cargo is full ({current_units}/{capacity}). No space for purchase.")

    # Determine target amount (cargo-limited)
    target_units = available_space if units is None else min(units, available_space)

    # Look up cached price + trade volume for this good
    market_cache = load_market_cache()
    max_per_transaction = target_units  # Default: try to buy full amount
    cached_price = None
    if waypoint in market_cache:
        for good in market_cache[waypoint].get("trade_goods", []):
            if good.get("symbol") == trade_symbol:
                vol = good.get("tradeVolume")
                if vol and vol > 0:
                    max_per_transaction = vol
                cached_price = good.get("purchasePrice")
                break

    # Cap by affordability using cached price + current credits
    agent = client.get_agent()
    credits = agent.get("credits") if isinstance(agent, dict) and "error" not in agent else None
    if credits is not None and cached_price and cached_price > 0:
        affordable = credits // cached_price
        if affordable <= 0:
            reason = f"Insufficient credits ({credits} cr, need {cached_price} cr/unit)."
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

    while total_purchased < target_units:
        # Refresh cargo to check current space
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" in cargo_data:
            break
        current_units = cargo_data.get("units", 0)
        available_space = capacity - current_units

        if available_space <= 0:
            break  # Cargo full

        units_to_buy = min(max_per_transaction, available_space, target_units - total_purchased)

        data = client.buy_cargo(ship_symbol, trade_symbol, units_to_buy)

        if isinstance(data, dict) and "error" in data:
            if purchase_count == 0:
                raise Exception(data['error'])
            stop_reason = data['error']
            break

        tx = data.get("transaction", {})
        units_bought = tx.get("units", units_to_buy)
        price_per_unit = tx.get("pricePerUnit", 0)
        total_purchased += units_bought
        total_cost += tx.get("totalPrice", 0)
        purchase_count += 1

        if max_price and price_per_unit > max_price:
            raise PriceCeilingHit(trade_symbol, price_per_unit, max_price, total_purchased, total_cost)

    # Check minimum quantity requirement
    if min_qty and total_purchased < min_qty:
        reason = stop_reason or f"cargo space or credits limited purchase to {total_purchased}"
        raise MinQtyNotMet(trade_symbol, total_purchased, min_qty, reason)

    # Get final cargo state
    cargo_data = client.get_cargo(ship_symbol)
    final_cargo = cargo_data.get("units", 0) if isinstance(cargo_data, dict) and "error" not in cargo_data else "?"

    if purchase_count > 1:
        return f"Purchased {total_purchased} {trade_symbol} for {total_cost} cr ({purchase_count} transactions). Cargo: {final_cargo}/{capacity}."
    else:
        return f"Purchased {total_purchased} {trade_symbol} for {total_cost} cr. Cargo: {final_cargo}/{capacity}."


def _deliver_contract_logic(contract_id: str, ship_symbol: str, trade_symbol: str, units: int = None) -> str:
    """
    Smart delivery: automatically calculates optimal units to deliver.
    Respects: contract requirements, cargo available, and explicit unit request.
    """
    # 1. Fetch contract to see remaining units needed
    contracts = client.list_contracts()
    if not isinstance(contracts, list):
        raise Exception("Could not fetch contracts")

    contract = None
    for c in contracts:
        if c.get("id") == contract_id:
            contract = c
            break

    if not contract:
        raise Exception(f"Contract {contract_id} not found")

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
        raise Exception(data['error'])

    # Return summary
    contract = data.get("contract", {})
    terms = contract.get("terms", {})
    result = f"Delivered {final_units} {trade_symbol} to contract {contract_id}."
    for d in terms.get("deliver", []):
        if d.get("tradeSymbol") == trade_symbol:
            result += f" Progress: {d.get('unitsFulfilled', 0)}/{d.get('unitsRequired', 0)}"
    return result


def _refuel_ship_logic(ship_symbol: str) -> str:
    _ensure_dock_logic(ship_symbol)
    data = client.refuel(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data['error'])

    fuel = data.get("fuel", {})
    tx = data.get("transaction", {})
    return f"Refueled {ship_symbol}. Fuel: {fuel.get('current')}/{fuel.get('capacity')}. Cost: {tx.get('totalPrice', '?')} cr."

def _transfer_cargo_logic(from_ship: str, to_ship: str, trade_symbol: str, units: int = None) -> str:
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
                raise Exception(data['error'])

            results.append(f"  {symbol}: {transfer_units} units")
            total_units += transfer_units

        if not results:
            return f"No cargo available to transfer from {from_ship}."

        return (
            f"Transferred all cargo from {from_ship} to {to_ship}:\n" +
            "\n".join(results) +
            f"\nTotal: {total_units} units transferred"
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
        raise Exception(data['error'])

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
    TRANSFER = "transfer"
    ALERT = "alert"
    REPEAT = "repeat"
    STOP = "stop"
    NEGOTIATE = "negotiate"
    BUY_SHIP = "buy_ship"

@dataclass
class Step:
    step_type: StepType
    args: list[str] = field(default_factory=list)
    def __str__(self):
        if self.args: return f"{self.step_type.value} {' '.join(self.args)}"
        return self.step_type.value

def _refresh_market_cache(waypoint: str) -> None:
    """Fetch live market data for a waypoint and update the cache. Silent on error."""
    system = "-".join(waypoint.split("-")[:2])
    try:
        data = client.get_market(system, waypoint)
        if data and isinstance(data, dict) and "error" not in data:
            _save_market_to_cache(waypoint, data)
    except Exception:
        pass

def parse_steps(steps_str: str) -> list[Step]:
    steps = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part: continue
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
        if not BEHAVIORS_FILE.exists(): return
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
                    except ValueError: continue
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
        try: self._last_mtime = BEHAVIORS_FILE.stat().st_mtime
        except OSError: pass

    def sync_state(self):
        if not BEHAVIORS_FILE.exists(): return
        try:
            if BEHAVIORS_FILE.stat().st_mtime > self._last_mtime:
                self.behaviors.clear()
                self._load()
        except OSError: pass

    def assign(self, ship_symbol: str, steps_str: str, start_step: int = 0) -> str:
        try:
            steps = parse_steps(steps_str)
        except ValueError as e: return f"Error: {e}"
        if not steps: return "Error: no steps provided"

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
            lines.append(f"  {cfg.ship_symbol}: step {step_idx + 1}/{len(cfg.steps)} [{current_step}] ({status})")
        return "\n".join(lines)

    def pause(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg: return f"{ship_symbol} has no assigned behavior."
        if cfg.paused: return f"{ship_symbol} is already paused."
        cfg.paused = True
        self._save()
        return f"Paused {ship_symbol} at step {cfg.current_step_index + 1}/{len(cfg.steps)}."

    def resume(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg or not cfg.paused: return "Nothing to resume."
        cfg.paused = False
        cfg.alert_sent = False
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        self._save()
        return f"Resumed {ship_symbol}."

    def skip_step(self, ship_symbol: str) -> str:
        cfg = self.behaviors.get(ship_symbol)
        if not cfg: return "No behavior."
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
        if not cfg or cfg.paused: return None

        ship = fleet.get_ship(ship_symbol)
        if not ship or not ship.is_available(): return None

        if cfg.current_step_index >= len(cfg.steps):
            cfg.current_step_index = 0
            cfg.step_phase = "INIT"

        step = cfg.steps[cfg.current_step_index]
        step_display = f"{step.step_type.value} {' '.join(step.args)}".strip()

        try:
            result = None
            if step.step_type == StepType.MINE: result = self._step_mine(cfg, step, ship, fleet)
            elif step.step_type == StepType.GOTO: result = self._step_goto(cfg, step, ship, fleet)
            elif step.step_type == StepType.BUY: result = self._step_buy(cfg, step, ship, fleet)
            elif step.step_type == StepType.SELL: result = self._step_sell(cfg, step, ship, fleet)
            elif step.step_type == StepType.REFUEL: result = self._step_refuel(cfg, step, ship, fleet)
            elif step.step_type == StepType.DELIVER: result = self._step_deliver(cfg, step, ship, fleet)
            elif step.step_type == StepType.TRANSFER: result = self._step_transfer(cfg, step, ship, fleet)
            elif step.step_type == StepType.SCOUT: result = self._step_scout(cfg, step, ship, fleet)
            elif step.step_type == StepType.REPEAT: result = self._step_repeat(cfg)
            elif step.step_type == StepType.STOP: result = self._step_stop(cfg)
            elif step.step_type == StepType.ALERT: result = self._step_alert(cfg, step)
            # Store last action for logging
            cfg.last_action = f"step {cfg.current_step_index + 1}: {step_display}"
            self._save()
            return result
        except Exception as e:
            cfg.error_message = str(e)
            cfg.paused = True
            cfg.last_action = f"step {cfg.current_step_index + 1}: {step_display} [ERROR: {str(e)}]"
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
            self._advance(cfg)
            return None

        # 2. INIT Phase: Prepare to move
        if cfg.step_phase == "INIT":
            # A. Check Refuel Needs (Critical for multi-hop intermediate stops)
            # If we are docked (likely from previous hop arrival) and need fuel
            if ship.fuel_capacity > 0:
                try:
                    # Heuristic: only refuel if we can't make a generic jump or are low
                    # But easiest is: if we are at a fuel market, just top up.
                    market_cache = load_market_cache()
                    curr_market = market_cache.get(ship.location, {})
                    # Check imports/exchange for FUEL
                    has_fuel = "FUEL" in curr_market.get("exchange", []) or "FUEL" in curr_market.get("exports", [])

                    if has_fuel and ship.fuel_current < ship.fuel_capacity:
                        _refuel_ship_logic(cfg.ship_symbol)
                except Exception:
                    pass # Ignore refuel errors (maybe no credits or no market), try to fly anyway

            # B. Navigate to next hop
            try:
                # _navigate_ship_logic handles the pathfinding for the NEXT hop
                # It returns the wait time for the IMMEDIATE hop, not total travel
                msg, wait = _navigate_ship_logic(cfg.ship_symbol, dest_wp, mode=mode)

                if wait > 0:
                    fleet.set_transit(cfg.ship_symbol, wait)
                    # IMPORTANT: Do NOT optimistically set location to dest_wp here.
                    # We might only be going to an intermediate stop.
                    # We will rely on API refresh in WAITING phase to update location.

                cfg.step_phase = "WAITING"
                self._save()
                return None
            except Exception as e:
                # If we get an error (e.g. stranded), pause and alert
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
                        self._advance(cfg)
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
                        client.jettison(cfg.ship_symbol, item["symbol"], item["units"])
            return None

    def _step_sell(self, cfg, step, ship, fleet) -> Optional[str]:
        target = step.args[0] if step.args else "*"
        # Parse min:PRICE from args
        min_price = None
        for arg in step.args[1:]:
            if arg.startswith("min:"):
                min_price = int(arg[4:])
                break
        # Refresh market cache before selling so prices are current
        if ship.location:
            _refresh_market_cache(ship.location)
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
                    if target != "*": raise e

        self._advance(cfg)
        return None

    def _step_buy(self, cfg, step, ship, fleet) -> Optional[str]:
        """Buy cargo from current market. Usage: buy TRADE_SYMBOL [UNITS] [max:PRICE] [min_qty:N]"""
        if not step.args:
            raise Exception("buy step requires trade symbol (e.g., 'buy IRON_ORE 10')")

        # Refresh market cache before buying so affordability check uses current prices
        if ship.location:
            _refresh_market_cache(ship.location)

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
            _buy_cargo_logic(cfg.ship_symbol, trade_symbol, units, max_price=max_price, min_qty=min_qty)
        except (PriceCeilingHit, MinQtyNotMet) as e:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: {e}"

        self._advance(cfg)
        return None

    def _step_refuel(self, cfg, step, ship, fleet) -> Optional[str]:
        if ship.fuel_capacity > 0:
            _refuel_ship_logic(cfg.ship_symbol)
        self._advance(cfg)
        return None

    def _step_deliver(self, cfg, step, ship, fleet) -> Optional[str]:
        """Deliver cargo for a contract. Usage: deliver CONTRACT_ID ITEM [UNITS]"""
        if len(step.args) < 2:
            raise Exception("deliver step requires contract_id and trade_symbol (e.g., 'deliver CONT001 DIAMONDS 5')")

        contract_id = step.args[0]
        trade_symbol = step.args[1]
        units = int(step.args[2]) if len(step.args) > 2 else None

        try:
            _deliver_contract_logic(contract_id, cfg.ship_symbol, trade_symbol, units)
        except Exception as e:
            raise e

        self._advance(cfg)
        return None

    def _step_transfer(self, cfg, step, ship, fleet) -> Optional[str]:
        """Transfer cargo to another ship. Usage: transfer DESTINATION_SHIP TRADE_SYMBOL [UNITS]

        TRADE_SYMBOL can be '*' to transfer all cargo. UNITS is optional (defaults to max available).
        Examples: transfer SHIP-2 IRON_ORE, transfer SHIP-2 IRON_ORE 50, transfer SHIP-2 *
        """
        if len(step.args) < 2:
            raise Exception("transfer step requires destination ship and trade symbol (e.g., 'transfer SHIP-2 IRON_ORE 50' or 'transfer SHIP-2 *')")

        destination_ship = step.args[0]
        trade_symbol = step.args[1]
        units = int(step.args[2]) if len(step.args) > 2 else None

        try:
            _transfer_cargo_logic(cfg.ship_symbol, destination_ship, trade_symbol, units)
        except Exception as e:
            raise e

        self._advance(cfg)
        return None

    def _step_scout(self, cfg, step, ship, fleet) -> Optional[str]:
        if not ship.location: raise Exception("No location")
        _refresh_market_cache(ship.location)
        self._advance(cfg)
        return None

    def _step_alert(self, cfg, step) -> Optional[str]:
        if not cfg.alert_sent:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            return f"{cfg.ship_symbol} ALERT: {' '.join(step.args)}"
        return None

    def _step_repeat(self, cfg) -> Optional[str]:
        cfg.current_step_index = 0
        cfg.step_phase = "INIT"
        self._save()
        return None

    def _step_stop(self, cfg) -> Optional[str]:
        """End the behavior and return ship to IDLE (manual control)."""
        self.cancel(cfg.ship_symbol)
        return None

    def _advance(self, cfg):
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        cfg.error_message = ""
        cfg.alert_sent = False
        self._save()


# ──────────────────────────────────────────────
#  Observation tools
# ──────────────────────────────────────────────

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
def view_contracts() -> str:
    """[READ-ONLY] List all contracts with status, terms, and delivery requirements."""
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
    """[READ-ONLY] List all ships with location, fuel, status, cooldowns, and cargo."""
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
    """[READ-ONLY] View the cargo contents of a specific ship."""
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
    """[READ-ONLY] View detailed ship info: mounts, modules, capabilities."""
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
def find_waypoints(
    waypoint_type: str = None,
    trait: str = None,
    trade_symbol: str = None,
    system_symbol: str = None,
    reference_ship: str = None
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
        ship = client.get_ship(reference_ship)
        if isinstance(ship, dict) and "error" not in ship:
            system_symbol = ship['nav']['systemSymbol']
            ref_coords = (ship['nav']['route']['destination']['x'], ship['nav']['route']['destination']['y'])

    if not system_symbol:
        # Fallback to agent HQ system
        agent = client.get_agent()
        if "headquarters" in agent:
            system_symbol = "-".join(agent["headquarters"].split("-")[:2])
        else:
            return "Error: system_symbol or reference_ship must be provided."

    # 2. Logic Branch: Trade Symbol Search (formerly query_markets/find_nearest)
    if trade_symbol:
        cache = load_market_cache()
        candidates = []

        # We need waypoint coordinates for the whole system to calculate distance
        all_wps = client.list_waypoints(system_symbol)
        wp_lut = {w['symbol']: (w['x'], w['y']) for w in all_wps} if isinstance(all_wps, list) else {}

        for wp_sym, mdata in cache.items():
            # Check if this market is in the target system
            if not wp_sym.startswith(system_symbol):
                continue

            # Check if good exists here
            goods = mdata.get('trade_goods', [])
            exports = mdata.get('exports', [])
            imports = mdata.get('imports', [])
            exchange = mdata.get('exchange', [])

            # Check price data first
            price_match = next((g for g in goods if g['symbol'] == trade_symbol), None)

            # Check structural data
            is_import = trade_symbol in imports
            is_export = trade_symbol in exports
            is_exchange = trade_symbol in exchange

            if price_match or is_import or is_export or is_exchange:
                dist = float('inf')
                if ref_coords and wp_sym in wp_lut:
                    wx, wy = wp_lut[wp_sym]
                    dist = math.sqrt((wx - ref_coords[0])**2 + (wy - ref_coords[1])**2)

                details = []
                if price_match:
                    if price_match.get('purchasePrice'):
                        details.append(f"BUY: {price_match['purchasePrice']}")
                    if price_match.get('sellPrice'):
                        details.append(f"SELL: {price_match['sellPrice']}")
                else:
                    if is_import:
                        details.append("Imports (Buy)")
                    if is_export:
                        details.append("Exports (Sell)")
                    if is_exchange:
                        details.append("Exchange")

                candidates.append((dist, wp_sym, ", ".join(details)))

        if not candidates:
            return f"No known markets for {trade_symbol} in {system_symbol}. (Note: Only checks cached data. Use scan_system or view_market to update cache)."

        # Sort
        candidates.sort(key=lambda x: x[0])
        lines = [f"Markets for {trade_symbol} in {system_symbol}:"]
        for dist, sym, det in candidates:
            d_str = f" ({dist:.1f} dist)" if dist != float('inf') else ""
            lines.append(f"  {sym}{d_str}: {det}")
        return "\n".join(lines)

    # 3. Logic Branch: Waypoint/Trait Search
    params = {}
    if waypoint_type:
        params['type'] = waypoint_type
    if trait:
        params['traits'] = trait

    data = client.list_waypoints(system_symbol, **params)

    # Special case: ASTEROID should also fetch ENGINEERED_ASTEROID
    if waypoint_type == "ASTEROID" and isinstance(data, list):
        eng = client.list_waypoints(system_symbol, type="ENGINEERED_ASTEROID")
        if isinstance(eng, list):
            data.extend(eng)

    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    if not data:
        return f"No waypoints found in {system_symbol} matching your criteria."

    # Sort by distance
    if ref_coords:
        def get_dist(w):
            return math.sqrt((w['x'] - ref_coords[0])**2 + (w['y'] - ref_coords[1])**2)
        data.sort(key=get_dist)

    lines = [f"Waypoints in {system_symbol}:"]
    for wp in data[:20]:  # Limit output
        d_str = ""
        if ref_coords:
            d = math.sqrt((wp['x'] - ref_coords[0])**2 + (wp['y'] - ref_coords[1])**2)
            d_str = f" ({d:.1f} dist)"

        t_list = [t['symbol'] for t in wp.get('traits', [])]
        lines.append(f"  {wp['symbol']} [{wp['type']}]{d_str} - {', '.join(t_list)}")

    if len(data) > 20:
        lines.append(f"  ... {len(data)-20} more ...")

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
    """View market prices and shipyard info at a waypoint. Always returns complete info:
    live prices if a ship is present, otherwise cached prices with staleness indicator.

    Args:
        waypoint_symbol: Waypoint with market (e.g., 'X1-KD26-B7')
    """
    import time
    system_symbol = '-'.join(waypoint_symbol.split('-')[:2])
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
        for section, label in [("exports", "Exports"), ("imports", "Imports"), ("exchange", "Exchange")]:
            items = api_data.get(section, [])
            if items:
                lines.append(f"  {label}: {', '.join(i['symbol'] if isinstance(i, dict) else i for i in items)}")
    elif cached:
        for section, label in [("exports", "Exports"), ("imports", "Imports"), ("exchange", "Exchange")]:
            items = cached.get(section, [])
            if items:
                lines.append(f"  {label}: {', '.join(items)}")

    # Price data — live if available, else cached with age
    if live_trade_goods:
        lines.append("  Prices (live):")
        for g in live_trade_goods:
            lines.append(f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} (vol: {g.get('tradeVolume', '?')})")
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
                lines.append(f"    {g['symbol']}: buy {g.get('purchasePrice', '?')} / sell {g.get('sellPrice', '?')} (vol: {g.get('tradeVolume', '?')})")
        else:
            lines.append("  No price data (no ship present, nothing cached).")
            if api_error:
                lines.append(f"  API error: {api_error}")

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
        pass

    return "\n".join(lines)


@tool
def query_markets(
    trade_symbol: Optional[str] = None,
    near_waypoint: Optional[str] = None
) -> str:
    """[READ-ONLY] Query cached market data with flexible filters.

    By default, returns all cached markets. Can filter by:
    - trade_symbol: Find markets that buy/sell/exchange a specific good (e.g., "IRON_ORE", "FUEL")
    - near_waypoint: Sort markets by distance from a waypoint (e.g., "X1-KD26-A1")

    Args:
        trade_symbol: Trade good to search for (e.g., "FUEL", "IRON_ORE")
        near_waypoint: Waypoint symbol to sort results by distance from

    Returns: Formatted list of matching markets with their trade goods and prices
    """
    import time

    cache = load_market_cache()

    if not cache:
        return "Market cache is empty. Use view_market to populate it."

    # Get reference point coordinates
    ref_x, ref_y = None, None
    # Cache waypoint lists per system to avoid redundant API calls
    waypoint_coords: dict = {}  # symbol -> (x, y)

    def get_waypoint_coords(system_symbol: str) -> dict:
        """Fetch and cache all waypoint coords for a system."""
        all_waypoints = client.list_waypoints(system_symbol)
        coords = {}
        if isinstance(all_waypoints, list):
            for wp in all_waypoints:
                coords[wp.get('symbol')] = (wp.get('x'), wp.get('y'))
        return coords

    if near_waypoint:
        system_symbol = '-'.join(near_waypoint.split('-')[:2])
        waypoint_coords.update(get_waypoint_coords(system_symbol))
        ref_x, ref_y = waypoint_coords.get(near_waypoint, (None, None))

    def dist_to_ref(x, y) -> float:
        if ref_x is None or x is None:
            return float('inf')
        return ((ref_x - x) ** 2 + (ref_y - y) ** 2) ** 0.5

    # Process each cached market, collecting (dist, waypoint_symbol, market_data, matching_goods)
    results = []
    for waypoint_symbol, market_data in cache.items():
        trade_goods = market_data.get('trade_goods', [])

        # Filter by trade_symbol if specified
        if trade_symbol:
            matching_goods = [g for g in trade_goods if g.get('symbol') == trade_symbol]
            if not matching_goods:
                continue
        else:
            matching_goods = trade_goods
            if not (market_data.get('exports') or market_data.get('imports') or market_data.get('exchange') or trade_goods):
                continue

        # Get coordinates for distance sorting
        dist = float('inf')
        if ref_x is not None:
            sys_sym = '-'.join(waypoint_symbol.split('-')[:2])
            if sys_sym not in {'-'.join(k.split('-')[:2]) for k in waypoint_coords}:
                waypoint_coords.update(get_waypoint_coords(sys_sym))
            x, y = waypoint_coords.get(waypoint_symbol, (None, None))
            dist = dist_to_ref(x, y)

        results.append((dist, waypoint_symbol, market_data, matching_goods))

    if not results:
        filters = []
        if trade_symbol:
            filters.append(f"trade_symbol={trade_symbol}")
        if near_waypoint:
            filters.append(f"near_waypoint={near_waypoint}")
        filter_str = " with filters: " + ", ".join(filters) if filters else ""
        return f"No markets found{filter_str}."

    # Sort by distance if reference point was given
    if ref_x is not None:
        results.sort(key=lambda r: r[0])

    def format_staleness(timestamp: int) -> str:
        now = int(time.time())
        age = now - timestamp
        if age < 60:
            return f"{age}s ago"
        elif age < 3600:
            return f"{age // 60}m ago"
        elif age < 86400:
            return f"{age // 3600}h ago"
        else:
            return f"{age // 86400}d ago"

    # Format results
    lines = []
    for dist, waypoint_symbol, market_data, goods in results:
        # Build header annotations
        annotations = []
        if dist != float('inf'):
            annotations.append(f"dist:{dist:.0f}")
        if market_data.get('last_updated') and any(g.get('purchasePrice') or g.get('sellPrice') for g in goods):
            annotations.append(format_staleness(market_data['last_updated']))
        header = waypoint_symbol
        if annotations:
            header += f" ({', '.join(annotations)})"
        lines.append(f"\n{header}:")

        # Show market structure (exports/imports/exchange) when not filtering by good
        if not trade_symbol:
            exports = market_data.get('exports', [])
            imports = market_data.get('imports', [])
            exchange = market_data.get('exchange', [])
            if exports:
                lines.append(f"  exports:  {', '.join(exports)}")
            if imports:
                lines.append(f"  imports:  {', '.join(imports)}")
            if exchange:
                lines.append(f"  exchange: {', '.join(exchange)}")

        # Show live prices if available
        priced_goods = [g for g in goods if g.get('purchasePrice') or g.get('sellPrice')]
        if priced_goods:
            lines.append("  prices:")
            for g in priced_goods:
                symbol = g.get('symbol', '?')
                buy_price = g.get('purchasePrice')
                sell_price = g.get('sellPrice')
                volume = g.get('tradeVolume')
                parts = []
                if buy_price:
                    parts.append(f"buy {buy_price}")
                if sell_price:
                    parts.append(f"sell {sell_price}")
                vol_str = f" vol:{volume}" if volume else ""
                lines.append(f"    {symbol}: {' / '.join(parts)}{vol_str}")

    return "\n".join(lines) if lines else "No market data available"


# ──────────────────────────────────────────────
#  Action tools
# ──────────────────────────────────────────────

@tool
def accept_contract(contract_id: str) -> str:
    """[STATE: credits, contract status] Accept a contract. Gives upfront credit payment."""
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
    """[STATE: credits, fleet] Purchase a ship at a shipyard. Need a ship present and enough credits."""
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
    """[STATE: nav_status] Put a ship into orbit. Required before navigating or extracting."""
    data = client.orbit(ship_symbol)
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    nav = data.get("nav", {})
    return f"{ship_symbol} is now in orbit at {nav.get('waypointSymbol', '?')}."

# ──────────────────────────────────────────────
#  Navigation & Action tools
# ──────────────────────────────────────────────

@tool
def navigate_ship(ship_symbol: str, destination_symbol: str, mode: str = "CRUISE") -> str:
    """
    [STATE: position, fuel] Smart Navigation.
    - If destination is close: Navigates directly.
    - If destination requires multi-hop: ENGAGES AUTOPILOT (assigns a temporary 'goto' behavior).
    The ship will automatically refuel and hop until it reaches the destination.
    """
    try:
        # 1. Check if this requires multi-hop
        ship = client.get_ship(ship_symbol)
        if isinstance(ship, dict) and "error" in ship:
            return f"Error: {ship['error']}"

        nav = ship.get("nav", {})
        current_wp = nav.get("waypointSymbol", "")
        if current_wp == destination_symbol:
            return f"{ship_symbol} is already at {destination_symbol}."

        # Fetch route details without executing to check distance/hops
        # We assume standard CRUISE for calculation if mode not specified
        waypoints = client.list_waypoints(nav.get("systemSymbol"))
        origin_obj = next((w for w in waypoints if w['symbol'] == current_wp), None)
        target_obj = next((w for w in waypoints if w['symbol'] == destination_symbol), None)

        if origin_obj and target_obj:
            path = _find_refuel_path(ship, origin_obj, target_obj, waypoints, mode)

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
                return (f"🚀 AUTOPILOT ENGAGED for {ship_symbol}. Multi-hop route detected ({stops} stops).\n"
                        f"Assigned behavior: '{cmds}'\n"
                        f"Ship will auto-refuel and travel to {destination_symbol}. check 'view_ships' for progress.")

        # 3. If direct route or simple jump, just execute standard logic
        msg, wait = _navigate_ship_logic(ship_symbol, destination_symbol, mode, execute=True)
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
def sell_cargo(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False) -> str:
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
def jettison_cargo(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False) -> str:
    """[STATE: cargo] Jettison cargo into space. Refuses to jettison contract goods unless force=True."""
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            return (
                f"Error: {trade_symbol} is required by an active contract. "
                f"Use deliver_contract to deliver it instead. "
                f"Call jettison_cargo with force=True to jettison anyway."
            )

    safe_units = None
    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        return f"Error checking cargo for {ship_symbol}: {cargo_data['error']}"

    inventory = cargo_data.get("inventory", [])
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        return f"Error: Ship {ship_symbol} has no {trade_symbol} available."

    if units is None:
        safe_units = available
    else:
        safe_units = min(units, available)

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
    lines = [f"Created {len(surveys)} survey(s). Cooldown: {cooldown.get('remainingSeconds', 0)}s"]
    for i, survey in enumerate(surveys):
        deposits = [d.get("symbol", "?") for d in survey.get("deposits", [])]
        lines.append(f"  Survey {i+1}: {survey.get('signature', '?')} - Size: {survey.get('size', '?')}")
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
    lines = [f"Found {len(waypoints)} waypoints. Cooldown: {cooldown.get('remainingSeconds', 0)}s"]
    for wp in waypoints[:15]:  # Limit output
        traits = ", ".join(t.get("symbol", "") for t in wp.get("traits", [])[:3])
        lines.append(f"  {wp.get('symbol', '?')} ({wp.get('type', '?')}) - {traits}")
    if len(waypoints) > 15:
        lines.append(f"  ... and {len(waypoints) - 15} more")
    return "\n".join(lines)


@tool
def scan_ships(ship_symbol: str) -> str:
    """[STATE: cooldown] Scan for other ships in the area. Ship must be in orbit."""
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
    """[STATE: position, cooldown] Jump to another star system via jump gate. Requires antimatter."""
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
    """[STATE: position, fuel] Warp to a waypoint in another system. Uses antimatter."""
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
    """[STATE: contracts] Negotiate a new contract. Ship must be docked at a faction HQ."""
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
    """[STATE: flight_mode] Set ship flight mode: CRUISE, BURN (2x fuel), DRIFT (1 fuel), STEALTH."""
    data = client.set_flight_mode(ship_symbol, mode.upper())
    if isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    return f"{ship_symbol} flight mode set to {mode.upper()}"


@tool
def deliver_contract(contract_id: str, ship_symbol: str, trade_symbol: str, units: int = None) -> str:
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
    agent = data.get("agent", {})
    return (
        f"Contract {contract_id} fulfilled!\n"
        f"Credits now: {agent.get('credits', '?')}"
    )


# ──────────────────────────────────────────────
#  Trade analysis
# ──────────────────────────────────────────────


@tool
def find_trades(ship_symbol: str = None, good: str = None, min_profit: int = 1) -> str:
    """[READ-ONLY] Find profitable trade routes from cached market data.

    Compares buy prices (exports/exchange) with sell prices (imports/exchange) across
    all known markets. Only uses cached price data — send ships to markets for fresh prices.

    Args:
        ship_symbol: Optional. If given, shows distance from this ship and sorts by profit/distance.
        good: Optional. Filter to a specific trade good (e.g. "IRON_ORE").
        min_profit: Minimum profit per unit to include (default: 1).
    """
    import time

    cache = load_market_cache()
    if not cache:
        return "No market data cached. Send ships to markets or run scan_system first."

    # Build per-good source/sink lists from markets that have price data
    # source = where to BUY (exports/exchange) — cost is sellPrice (what you pay the market)
    # sink = where to SELL (imports/exchange) — revenue is purchasePrice (what market pays you)
    sources = {}  # good -> [(market_wp, buy_cost, volume)]
    sinks = {}    # good -> [(market_wp, sell_revenue, volume)]

    for wp, mdata in cache.items():
        trade_goods = mdata.get("trade_goods")
        if not trade_goods:
            continue

        exports = set(mdata.get("exports", []))
        imports = set(mdata.get("imports", []))
        exchange = set(mdata.get("exchange", []))
        source_goods = exports | exchange
        sink_goods = imports | exchange

        for tg in trade_goods:
            sym = tg["symbol"]
            buy_cost = tg.get("purchasePrice")       # what you PAY the market
            sell_revenue = tg.get("sellPrice")  # what market PAYS you

            if sym in source_goods and buy_cost is not None:
                sources.setdefault(sym, []).append((wp, buy_cost, tg.get("tradeVolume", 0)))
            if sym in sink_goods and sell_revenue is not None:
                sinks.setdefault(sym, []).append((wp, sell_revenue, tg.get("tradeVolume", 0)))

    # Filter to specific good if requested
    if good:
        good = good.upper()
        all_goods = {good} if good in sources and good in sinks else set()
    else:
        all_goods = set(sources.keys()) & set(sinks.keys())

    if not all_goods:
        if good:
            return f"No trade routes found for {good}. Need both a source (exports/exchange) and sink (imports/exchange) with price data."
        return "No trade routes found. Need more market price data — send ships to scout markets."

    # Build routes: pair each source with each sink, compute profit
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
                # Check staleness of both markets
                src_updated = cache.get(src_wp, {}).get("last_updated", 0)
                snk_updated = cache.get(snk_wp, {}).get("last_updated", 0)
                oldest = min(src_updated, snk_updated)
                stale = (now - oldest) > 7200 if oldest else True  # >2h
                routes.append({
                    "good": sym,
                    "src": src_wp,
                    "snk": snk_wp,
                    "buy": buy_cost,
                    "sell": sell_rev,
                    "profit": profit,
                    "volume": volume,
                    "stale": stale,
                })

    if not routes:
        return f"No profitable routes found (min_profit={min_profit}). Try lowering min_profit or scouting more markets."

    # If ship_symbol given, compute distance from ship to source and sort by efficiency
    ship_wp = None
    ship_pos = None
    wp_coords = {}

    if ship_symbol:
        ship = client.get_ship(ship_symbol)
        if isinstance(ship, dict) and "error" not in ship:
            nav = ship.get("nav", {})
            ship_wp = nav.get("waypointSymbol", "")
            system_symbol = nav.get("systemSymbol", "")

            waypoints = client.list_waypoints(system_symbol)
            if isinstance(waypoints, list):
                for wp in waypoints:
                    wp_coords[wp["symbol"]] = (wp.get("x", 0), wp.get("y", 0))
                ship_pos = wp_coords.get(ship_wp)

    if ship_pos:
        # Add distance info and sort by profit / distance (efficiency)
        for r in routes:
            src_pos = wp_coords.get(r["src"])
            if src_pos:
                dist = math.sqrt((src_pos[0] - ship_pos[0])**2 + (src_pos[1] - ship_pos[1])**2)
                r["dist"] = max(dist, 1.0)  # avoid div by zero
            else:
                r["dist"] = None
        routes.sort(key=lambda r: r["profit"] / r["dist"] if r.get("dist") else 0, reverse=True)
    else:
        # Sort by raw profit per unit
        routes.sort(key=lambda r: r["profit"], reverse=True)

    # Format top 10
    lines = ["Top trade routes (from cached prices):\n"]
    for r in routes[:10]:
        lines.append(f"  {r['good']}: buy at {r['src']} ({r['buy']}/unit) -> sell at {r['snk']} ({r['sell']}/unit)")
        detail = f"    Profit: {r['profit']}/unit | Volume: {r['volume']}"
        if ship_pos and r.get("dist"):
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
    """[STATE: behavior] Resume a paused behavior after handling an alert. Advances past the alert step."""
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
def assign_mining_loop(ship_symbol: str, asteroid_wp: str, ore_types: str = "", start_step: int = 0) -> str:
    """[STATE: behavior] Convenience: assign a mine-sell loop. Builds a step sequence internally.

    Args:
        ship_symbol: Ship to assign.
        asteroid_wp: Asteroid waypoint to mine at.
        ore_types: Comma-separated ore symbols to KEEP (e.g. "IRON_ORE,COPPER_ORE").
    """
    ore_list = [s.strip() for s in ore_types.split(",") if s.strip()] if ore_types else []
    ore_str = " ".join(ore_list) if ore_list else ""
    mine_part = f"mine {asteroid_wp} {ore_str}".strip()
    # Simple mine loop: mine until full, then alert for LLM to handle selling
    steps_str = f"{mine_part}, alert cargo full, repeat"
    return get_engine().assign(ship_symbol, steps_str, start_step)


@tool
def assign_trade_route(ship_symbol: str, buy_waypoint: str, buy_good: str, sell_waypoint: str, sell_good: str = None, start_step: int = 0) -> str:
    """[STATE: behavior] Assign a permanent buying and selling loop.
    The ship will:
    1. Go to buy_waypoint
    2. Buy the specified good (attempt to fill cargo) (automatically sets max price)
    3. Go to sell_waypoint
    4. Sell the good (automatically sets min price)
    5. Repeat until the market prices make the trade unprofitable.
    Smart navigation handles refueling automatically.
    Args:
        ship_symbol: The hauler ship.
        buy_waypoint: Where to buy (e.g., 'X1-KD26-D44').
        buy_good: The symbol to trade (e.g., 'SHIP_PARTS').
        sell_waypoint: Where to sell.
        sell_good: Optional. Defaults to buy_good. Use if refining/transforming, otherwise leave empty.
    """
    s_good = sell_good if sell_good else buy_good
    cache = load_market_cache()

    # Buy step: alert if price spikes more than 10% above the expected purchase price
    buy_step = f"buy {buy_good}"
    buy_market = cache.get(buy_waypoint, {})
    for good in buy_market.get("trade_goods", []):
        if good.get("symbol") == buy_good:
            buy_cost = good.get("purchasePrice")
            if buy_cost:
                max_buy = int(buy_cost * 1.10)
                buy_step = f"buy {buy_good} max:{max_buy}"
            break

    # Sell step: alert if price drops more than 10% below the expected sell price
    sell_step = f"sell {s_good}"
    sell_market = cache.get(sell_waypoint, {})
    for good in sell_market.get("trade_goods", []):
        if good.get("symbol") == s_good:
            sell_price = good.get("sellPrice")
            if sell_price:
                min_sell = int(sell_price * 0.90)
                sell_step = f"sell {s_good} min:{min_sell}"
            break

    steps_str = f"goto {buy_waypoint}, {buy_step}, goto {sell_waypoint}, {sell_step}, repeat"
    return get_engine().assign(ship_symbol, steps_str, start_step)


@tool
def assign_satellite_scout(ship_symbols: str, market_waypoints: str = "") -> str:
    """[STATE: behavior] Convenience: assign scouting to satellites.
    Args:
        ship_symbols: Comma-separated satellite ship symbols.
        market_waypoints: Comma-separated waypoints to scout.
                          IF OMITTED, uses ALL known marketplaces in the ship's system (from cache).
    """
    ships = [s.strip() for s in ship_symbols.split(",") if s.strip()]
    if not ships:
        return "Error: no ship symbols provided."

    engine = get_engine()
    markets = []

    # Case 1: User provided specific list
    if market_waypoints:
        markets = [m.strip() for m in market_waypoints.split(",") if m.strip()]

    # Case 2: Use Cache (Filtered by System)
    else:
        # Determine system from first ship
        first_ship = client.get_ship(ships[0])
        if isinstance(first_ship, dict) and "error" not in first_ship:
            system_symbol = first_ship['nav']['systemSymbol']
            cache = load_market_cache()

            # Filter cache keys for this system
            markets = [wp for wp in cache.keys() if wp.startswith(system_symbol)]
            markets.sort()

            if not markets:
                return f"Error: No markets found in cache for system {system_symbol}. Run scan_system first to populate cache."

    if not markets:
        return "Error: no markets found to scout."

    # Build one shared sequence: goto M1, scout, goto M2, scout, ..., repeat
    parts = []
    for mkt in markets:
        parts.append(f"goto {mkt}")
        parts.append("scout")
    parts.append("repeat")

    steps_str = ", ".join(parts)

    # Calculate offsets to spread ships out
    m = len(markets)
    already_placed = sum(
        1 for cfg in engine.behaviors.values()
        if cfg.steps_str == steps_str and cfg.ship_symbol not in ships
    )

    total = already_placed + len(ships)
    results = []
    for j, ship in enumerate(ships):
        slot = already_placed + j
        market_offset = (slot * m) // total
        start_step = market_offset * 2
        result = engine.assign(ship, steps_str, start_step=start_step)
        results.append(result)

    return "\n".join(results)


# ──────────────────────────────────────────────
#  Tool registry
# ──────────────────────────────────────────────

# Tier 1: Essential tools
TIER_1_TOOLS = [
    # Navigation & planning
    navigate_ship, plan_route,
    # Trading & cargo
    buy_cargo, sell_cargo, transfer_cargo, jettison_cargo,
    # Contracts
    accept_contract, deliver_contract, fulfill_contract, negotiate_contract,
    # Info
    view_market, view_ships, view_contracts, find_trades,
    # Locator
    find_waypoints,
    # Planning
    update_plan,
    # Behavior control
    resume_behavior, skip_step, cancel_behavior, pause_behavior,
    assign_mining_loop, assign_satellite_scout, assign_trade_route,
    create_behavior,
]

# All tools (tier 2)
ALL_TOOLS = [
    # Observation
    view_agent, view_contracts, view_ships, view_ship_details, view_cargo,
    scan_system, view_market, view_jump_gate,
    # Locator
    find_waypoints,
    # Ship operations
    orbit_ship, dock_ship, navigate_ship, refuel_ship, plan_route,
    # Mining & resources
    extract_ore, survey_asteroid,
    # Trading & cargo
    buy_cargo, sell_cargo, jettison_cargo, transfer_cargo,
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
    # Analysis
    find_trades,
    # Behavior control
    create_behavior, resume_behavior, skip_step, cancel_behavior, pause_behavior,
    assign_mining_loop, assign_satellite_scout, assign_trade_route,
]

# Tools that are "significant" actions worth narrating
SIGNIFICANT_TOOLS = {
    "extract_ore", "survey_asteroid", "navigate_ship", "buy_cargo", "sell_cargo", "jettison_cargo",
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
