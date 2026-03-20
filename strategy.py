"""
Fleet strategy and autonomous assignment planning.

Core functions for evaluating game phase, ship assignments, and intelligent scouting plans.
Depends on fleet state and cache, but no execution logic (that stays in tools.py).
"""

import logging
import time
from typing import Optional

from cache import (
    load_waypoint_cache,
    get_system_from_waypoint,
    _fetch_and_cache_construction,
    calculate_distance,
)

log = logging.getLogger(__name__)

# Injected globals (set by tools.py)
_fleet = None
_hq_managed_ships = "NONE"

# Constants (also defined in tools.py)
RESERVE_BUFFER = 350_000
GATE_MIN_CREDIT_BUFFER = 250_000

def set_fleet(fleet):
    """Set the fleet tracker reference."""
    global _fleet
    _fleet = fleet

def set_hq_managed_ships(ships: str):
    """Set which ships are HQ-managed."""
    global _hq_managed_ships
    _hq_managed_ships = ships


# ── These are imported from tools.py (lazy imports to avoid circular deps)
def _get_local_agent():
    """Stub - will be imported from tools when needed."""
    from tools import _get_local_agent as impl
    return impl()

def _get_local_ship(ship_symbol: str):
    """Stub - will be imported from tools when needed."""
    from tools import _get_local_ship as impl
    return impl(ship_symbol)

def get_hq_enabled():
    """Stub - will be imported from tools when needed."""
    from tools import get_hq_enabled as impl
    return impl()

def is_ship_hq_managed(ship_symbol: str, fleet):
    """Stub - will be imported from tools when needed."""
    from tools import is_ship_hq_managed as impl
    return impl(ship_symbol, fleet)


def evaluate_fleet_strategy(system_symbol: str | None = None) -> dict:
    """Core logic for game phase, budget, and fleet needs. Single source of truth."""
    agent = _get_local_agent()
    credits = agent.get("credits", 0)

    # Read instantly from local memory instead of API
    ships = list(_fleet.ships.values()) if _fleet else []

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


def _analyze_trade_routes(ship_symbol: str | None = None, min_profit: int = 1) -> list[dict]:
    """Helper: returns a list of trade dictionaries sorted by profitability."""
    from cache import load_market_cache

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
            from cache import get_system_waypoints
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


def _get_probe_plan(ship_symbol: str, ship_location: str, phase: int, claimed_targets: set | None = None, active_probe_systems: dict | None = None) -> str:
    """
    Determines a probe's mission using Time-Adjusted Staleness and Cluster Tours.

    1. Seed Selection: Uses (Age - FlightTime) to prioritize globally oldest markets.
    2. Cluster Tour: Once an urgent "Seed" is picked, adds nearby stale neighbors.
    """
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
    min_age = 600 if phase >= 4 else 300  # 10 mins in phase 4, 5 mins in phase 1-3

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
