"""
Unified waypoint, market, and shipyard caching layer.

Provides persistent cache for game data with automatic migration from legacy formats.
Includes waypoint fetching, market data storage, and distance calculations.
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

from api_client import SpaceTradersClient

log = logging.getLogger(__name__)

# Get API client (injected from outside or create locally)
client = None

def set_client(c: SpaceTradersClient):
    """Set the API client for cache operations."""
    global client
    client = c


WAYPOINT_CACHE_FILE = Path("waypoint_cache.json")
MARKET_CACHE_FILE = WAYPOINT_CACHE_FILE  # Alias for existing code


def get_system_from_waypoint(waypoint_symbol: str) -> str:
    """Extract the system symbol from a waypoint symbol (e.g., 'X1-ABC-123' -> 'X1-ABC')."""
    return waypoint_symbol.rsplit("-", 1)[0]


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def waypoint_distance(wp1_sym: str, wp2_sym: str, cache: Optional[dict] = None) -> float:
    """Calculate distance between two waypoints using cached coordinates."""
    if cache is None:
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
    if not client:
        return
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


def get_system_waypoints(system_symbol: str, waypoint_type: str | None = None, trait: str | None = None) -> list[dict]:
    """Get waypoints for a system, using cache if fully fetched, else API.

    On first call for a system, fetches all waypoints from API and caches them.
    Subsequent calls return cached data without API calls.

    Args:
        system_symbol: The system to fetch waypoints for
        waypoint_type: Optional filter by type (e.g., "JUMP_GATE", "ASTEROID")
        trait: Optional filter by trait (e.g., "MARKETPLACE", "SHIPYARD")
    """
    if not client:
        return []
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
