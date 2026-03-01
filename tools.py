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

    # Structural data
    for section in ("exports", "imports", "exchange"):
        items = data.get(section, [])
        if items:
            entry[section] = [i["symbol"] for i in items]

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
    dest_sys = "-".join(destination_symbol.split("-")[:2])
    target_wp_symbol = destination_symbol
    is_inter_system = False

    if current_sys != dest_sys:
        is_inter_system = True
        waypoints = client.list_waypoints(current_sys, traits="JUMP_GATE")
        if not waypoints:
             raise Exception(f"Destination is in {dest_sys}, but no Jump Gate found in {current_sys}.")
        target_wp_symbol = waypoints[0]['symbol']
        if current_wp == target_wp_symbol:
            return f"Ship is at Jump Gate. Use 'jump_ship' to jump to {dest_sys}.", 0.0

    # Fetch Waypoints
    waypoints = client.list_waypoints(current_sys)
    if isinstance(waypoints, dict) and "error" in waypoints:
        raise Exception(waypoints['error'])

    origin_obj = next((w for w in waypoints if w['symbol'] == current_wp), None)
    target_obj = next((w for w in waypoints if w['symbol'] == target_wp_symbol), None)
    if not origin_obj or not target_obj:
        raise Exception("Could not resolve waypoint coordinates.")

    # Execute Logic
    if execute:
        _, direct_cost, direct_time = _calculate_travel_cost(ship, target_obj, origin_obj, mode)
        fuel_available = fuel.get("current", 0)
        next_hop = target_wp_symbol
        is_multi_hop = False

        if fuel.get("capacity", 0) > 0 and direct_cost > fuel_available:
            # Try Refuel Here
            market_cache = load_market_cache()
            curr_market = market_cache.get(current_wp, {})
            has_fuel_here = "FUEL" in curr_market.get("exchange", []) or "FUEL" in curr_market.get("exports", [])

            if has_fuel_here:
                _ensure_dock_logic(ship_symbol)
                client.refuel(ship_symbol)
                ship = client.get_ship(ship_symbol) # update fuel state
                fuel_available = ship.get("fuel", {}).get("current", 0)

            # Route Plan if still needed
            if direct_cost > fuel_available:
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
        result = f"🚀 {ship_symbol} navigating to {next_hop} ({mode}). Est: {direct_time}s."

        if is_multi_hop:
            result += f"\nNote: Multi-hop route initiated. Stopping at {next_hop} to refuel."
        elif is_inter_system:
            result += f"\nNote: Arriving at Jump Gate. Use 'jump_ship' next."

        return result, wait_secs

    else:
        # Planning Mode (Read Only)
        # ... (simplified from previous tools.py for brevity) ...
        dist, _, _ = _calculate_travel_cost(ship, target_obj, origin_obj, "CRUISE")
        return f"Plan: {current_wp} -> {target_wp_symbol} (Dist: {dist})", 0.0

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
    if isinstance(data, dict) and "error" in data:
        # Handle API error 4000 (Cooldown) specifically if API returns it as error
        if "cooldown" in str(data['error']).lower():
             # Fallback guess
             return "Hit cooldown", 70.0
        raise Exception(data['error'])

    extraction = data.get("extraction", {})
    cd = data.get("cooldown", {})
    cargo = data.get("cargo", {})
    yielded = extraction.get("yield", {})

    result = f"Extracted {yielded.get('units', '?')} {yielded.get('symbol', '?')}."
    result += f" Cargo: {cargo.get('units', 0)}/{cargo.get('capacity', '?')}."

    return result, float(cd.get("remainingSeconds", 0))

def _sell_cargo_logic(ship_symbol: str, trade_symbol: str, units: int = None, force: bool = False) -> str:
    # 1. Check Contract Safety
    if not force:
        contract_goods = _get_contract_goods()
        if trade_symbol in contract_goods:
            raise Exception(f"{trade_symbol} is required by an active contract. Use force=True to override.")

    # 2. Check Availability
    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo: {cargo_data['error']}")

    inventory = cargo_data.get("inventory", [])
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        raise Exception(f"Ship {ship_symbol} has no {trade_symbol}.")

    final_units = available if units is None else min(units, available)

    # 3. Action
    _ensure_dock_logic(ship_symbol)
    data = client.sell_cargo(ship_symbol, trade_symbol, final_units)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data['error'])

    tx = data.get("transaction", {})
    cargo = data.get("cargo", {})
    return f"Sold {tx.get('units', final_units)} {trade_symbol} for {tx.get('totalPrice', '?')} cr. Cargo: {cargo.get('units')}/{cargo.get('capacity')}."

def _buy_cargo_logic(ship_symbol: str, trade_symbol: str, units: int = None) -> str:
    """Buy cargo from the current market. Returns (message, total_cost)."""
    # 1. Check available cargo space
    cargo_data = client.get_cargo(ship_symbol)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        raise Exception(f"Error checking cargo: {cargo_data['error']}")

    capacity = cargo_data.get("capacity", 0)
    current_units = cargo_data.get("units", 0)
    available_space = capacity - current_units

    if available_space <= 0:
        raise Exception(f"Ship {ship_symbol} cargo is full ({current_units}/{capacity}). No space for purchase.")

    # 2. Determine units to purchase (default to fill remaining space)
    final_units = available_space if units is None else min(units, available_space)

    # 3. Action
    _ensure_dock_logic(ship_symbol)
    data = client.buy_cargo(ship_symbol, trade_symbol, final_units)
    if isinstance(data, dict) and "error" in data:
        raise Exception(data['error'])

    tx = data.get("transaction", {})
    cargo = data.get("cargo", {})
    return f"Purchased {tx.get('units', final_units)} {trade_symbol} for {tx.get('totalPrice', '?')} cr. Cargo: {cargo.get('units')}/{cargo.get('capacity')}."


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
        if not self.behaviors: return "(no behaviors assigned -- all ships idle)"
        lines = []
        for cfg in self.behaviors.values():
            step_idx = cfg.current_step_index
            current_step = cfg.steps[step_idx] if step_idx < len(cfg.steps) else "?"
            status = "PAUSED" if cfg.paused else cfg.step_phase
            if cfg.error_message: status = f"ERROR: {cfg.error_message}"
            lines.append(f"  {cfg.ship_symbol}: step {step_idx + 1}/{len(cfg.steps)} [{current_step}] ({status})")
        return "\n".join(lines)

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
        """Smart navigation using _navigate_ship_logic."""
        dest_wp = step.args[0]

        if cfg.step_phase == "INIT":
            if ship.location == dest_wp and ship.nav_status != "IN_TRANSIT":
                self._advance(cfg)
                return None

            try:
                # Calls the SMART logic (auto-refuel, etc)
                msg, wait = _navigate_ship_logic(cfg.ship_symbol, dest_wp)
                if wait > 0:
                    fleet.set_transit(cfg.ship_symbol, wait)
                    ship.location = dest_wp # Optimistic update
                cfg.step_phase = "WAITING"
                self._save()
            except Exception as e:
                raise e # Caught by tick
            return None

        if cfg.step_phase == "WAITING":
            if ship.location == dest_wp or ship.nav_status != "IN_TRANSIT":
                self._advance(cfg)
            return None

    def _step_mine(self, cfg, step, ship, fleet) -> Optional[str]:
        """Smart mining using _extract_ore_logic."""
        asteroid_wp = step.args[0] if step.args else None
        ore_types = step.args[1:] if len(step.args) > 1 else []

        if cfg.step_phase == "INIT":
            if asteroid_wp and ship.location != asteroid_wp:
                # Reuse navigate logic logic dynamically
                msg, wait = _navigate_ship_logic(cfg.ship_symbol, asteroid_wp)
                if wait > 0:
                    fleet.set_transit(cfg.ship_symbol, wait)
                # We don't change phase to WAITING here because we want to loop back to INIT
                # until we are actually at the location.
                # Better: Use a sub-state or just let the next tick handle transit cooldown.
                return None

            cfg.step_phase = "EXTRACTING"
            self._save()
            return None

        if cfg.step_phase == "EXTRACTING":
            if ship.cargo_capacity > 0 and ship.cargo_units >= ship.cargo_capacity:
                # Alert operator when cargo is full — don't silently advance
                cfg.paused = True
                cfg.alert_sent = True
                self._save()
                return f"{cfg.ship_symbol} ALERT: Cargo full at {ship.location}. Define a sell/deliver step or manually empty cargo."

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
        # Get inventory
        c = client.get_cargo(cfg.ship_symbol)
        inventory = c.get("inventory", [])

        sold_something = False
        for item in inventory:
            sym = item["symbol"]
            if target == "*" or sym == target:
                try:
                    # Logic function handles contract checks
                    _sell_cargo_logic(cfg.ship_symbol, sym)
                    sold_something = True
                except Exception as e:
                    # If target is *, ignore failures (e.g. contract goods)
                    if target != "*": raise e

        self._advance(cfg)
        return None

    def _step_buy(self, cfg, step, ship, fleet) -> Optional[str]:
        """Buy cargo from current market. Usage: buy TRADE_SYMBOL [UNITS]"""
        if not step.args:
            raise Exception("buy step requires trade symbol (e.g., 'buy IRON_ORE 10')")

        trade_symbol = step.args[0]
        units = int(step.args[1]) if len(step.args) > 1 else None

        try:
            _buy_cargo_logic(cfg.ship_symbol, trade_symbol, units)
        except Exception as e:
            raise e

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

    def _step_scout(self, cfg, step, ship, fleet) -> Optional[str]:
        if not ship.location: raise Exception("No location")
        sys = "-".join(ship.location.split("-")[:2])
        m = client.get_market(sys, ship.location)
        if m and "error" not in m:
            _save_market_to_cache(ship.location, m)
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
    - If destination is in another system: Routes to the Jump Gate.
    - If destination is too far for one tank: Finds intermediate stops to refuel.
    """
    try:
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

    # Check source ship inventory
    cargo_data = client.get_cargo(from_ship)
    if isinstance(cargo_data, dict) and "error" in cargo_data:
        return f"Error checking cargo for {from_ship}: {cargo_data['error']}"

    inventory = cargo_data.get("inventory", [])
    available = 0
    for item in inventory:
        if item.get("symbol") == trade_symbol:
            available = item.get("units", 0)
            break

    if available == 0:
        return f"Error: Ship {from_ship} has no {trade_symbol} available."

    safe_units = available if units is None else min(units, available)

    # Ensure orbit states
    try:
        _ensure_orbit_logic(from_ship)
        _ensure_orbit_logic(to_ship)
    except Exception as e:
        return f"Error: {e}"

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
            buy_cost = tg.get("sellPrice")       # what you PAY the market
            sell_revenue = tg.get("purchasePrice")  # what market PAYS you

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
      buy ITEM [UNITS]           - Buy cargo from current market (fills remaining space by default)
      sell ITEM or sell *         - Sell cargo at current market (skips contract goods)
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
def assign_mining_loop(ship_symbol: str, asteroid_wp: str, ore_types: str = "") -> str:
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
    return get_engine().assign(ship_symbol, steps_str)


@tool
def assign_satellite_scout(ship_symbols: str, market_waypoints: str = "") -> str:
    """[STATE: behavior] Convenience: assign scouting to satellites. Builds step sequences internally.

    All satellites share the same step sequence but start at evenly-spaced offsets so
    they spread across the market list instead of converging on the first waypoint.

    Args:
        ship_symbols: Comma-separated satellite ship symbols (e.g. "WHATER-2,WHATER-3").
        market_waypoints: Comma-separated waypoints to scout. If omitted, uses all known markets.
    """
    ships = [s.strip() for s in ship_symbols.split(",") if s.strip()]

    if not ships:
        return "Error: no ship symbols provided."

    engine = get_engine()

    # Get market list
    if market_waypoints:
        markets = [m.strip() for m in market_waypoints.split(",") if m.strip()]
    else:
        cache = load_market_cache()
        markets = sorted(cache.keys())

    if not markets:
        return "Error: no markets known. Run scan_system first."

    # Build one shared sequence: goto M1, scout, goto M2, scout, ..., repeat
    parts = []
    for mkt in markets:
        parts.append(f"goto {mkt}")
        parts.append("scout")
    parts.append("repeat")
    steps_str = ", ".join(parts)

    m = len(markets)

    # Count ships already running this exact sequence (from previous calls).
    # This makes sequential single-ship calls spread out just like a batch call.
    already_placed = sum(
        1 for cfg in engine.behaviors.values()
        if cfg.steps_str == steps_str and cfg.ship_symbol not in ships
    )
    total = already_placed + len(ships)

    results = []
    for j, ship in enumerate(ships):
        slot = already_placed + j        # absolute position in the full fleet
        market_offset = (slot * m) // total
        start_step = market_offset * 2  # 2 steps per market (goto + scout)
        result = engine.assign(ship, steps_str, start_step=start_step)
        results.append(result)

    return "\n".join(results)


# ──────────────────────────────────────────────
#  Tool registry
# ──────────────────────────────────────────────

# Tier 1: Essential tools for mining/selling/contracts gameplay
# Tools auto-handle dock/orbit, so those are excluded
TIER_1_TOOLS = [
    # Navigation & planning
    navigate_ship, refuel_ship, plan_route,
    # Trading & cargo
    buy_cargo, sell_cargo, transfer_cargo, jettison_cargo,
    # Contracts
    accept_contract, deliver_contract, fulfill_contract, negotiate_contract,
    # Info (view_market includes shipyard data)
    scan_system, view_market, view_ships, find_waypoints, view_contracts, find_trades,
    # Planning
    update_plan,
    # Behavior control (step-sequence engine)
    create_behavior, resume_behavior, skip_step, cancel_behavior,
    assign_mining_loop, assign_satellite_scout,
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
    # Find stuff
    find_nearest, find_trades,
    # Behavior control (step-sequence engine)
    create_behavior, resume_behavior, skip_step, cancel_behavior,
    assign_mining_loop, assign_satellite_scout,
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
