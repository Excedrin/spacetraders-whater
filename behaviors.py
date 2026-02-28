"""
behaviors.py — Ship behavior state machines.

Ships can be assigned autonomous behaviors (loops) that run without LLM
involvement. The LLM acts as Fleet Admiral: assigning behaviors and only
being invoked when a ship is IDLE or a behavior raises an ALERT.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BEHAVIORS_FILE = Path("behaviors.json")

# Behavior types
BEHAVIOR_MINING_LOOP = "MINING_LOOP"
BEHAVIOR_SATELLITE_SCOUT = "SATELLITE_SCOUT"

# Shared states
STATE_START = "START"
STATE_ERROR = "ERROR"

# MINING_LOOP states
STATE_EXTRACT = "EXTRACT"
STATE_CARGO_FULL = "CARGO_FULL"

# SATELLITE_SCOUT states
STATE_NAVIGATE = "NAVIGATE"
STATE_SCOUT = "SCOUT"


@dataclass
class BehaviorConfig:
    ship_symbol: str
    behavior_type: str
    parameters: dict = field(default_factory=dict)
    current_state: str = STATE_START
    error_message: str = ""


_engine_instance: Optional["BehaviorEngine"] = None


def get_engine() -> "BehaviorEngine":
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BehaviorEngine()
    return _engine_instance


class BehaviorEngine:
    """
    Manages autonomous ship behaviors.

    Call tick() for each ship every loop. It advances the ship's state
    machine one step, returning an alert string when LLM intervention
    is required, or None if everything is running smoothly.
    """

    def __init__(self):
        self.behaviors: dict[str, BehaviorConfig] = {}
        self._load()

    def _load(self):
        if BEHAVIORS_FILE.exists():
            try:
                entries = json.loads(BEHAVIORS_FILE.read_text())
                for e in entries:
                    cfg = BehaviorConfig(
                        ship_symbol=e["ship_symbol"],
                        behavior_type=e["behavior_type"],
                        parameters=e.get("parameters", {}),
                        current_state=e.get("current_state", STATE_START),
                        error_message=e.get("error_message", ""),
                    )
                    self.behaviors[cfg.ship_symbol] = cfg
            except Exception:
                pass

    def _save(self):
        BEHAVIORS_FILE.write_text(json.dumps(
            [
                {
                    "ship_symbol": c.ship_symbol,
                    "behavior_type": c.behavior_type,
                    "parameters": c.parameters,
                    "current_state": c.current_state,
                    "error_message": c.error_message,
                }
                for c in self.behaviors.values()
            ],
            indent=2,
        ))

    def assign(self, ship_symbol: str, behavior_type: str, parameters: dict) -> None:
        """Assign a behavior to a ship, resetting its state."""
        self.behaviors[ship_symbol] = BehaviorConfig(
            ship_symbol=ship_symbol,
            behavior_type=behavior_type,
            parameters=parameters,
            current_state=STATE_START,
        )
        self._save()

    def cancel(self, ship_symbol: str) -> None:
        """Remove a ship's behavior. It returns to IDLE (LLM control)."""
        self.behaviors.pop(ship_symbol, None)
        self._save()

    def get_idle_ships(self, fleet) -> list[str]:
        """Ships with no assigned behavior — need LLM attention."""
        return [s for s in fleet.ships if s not in self.behaviors]

    def summary(self) -> str:
        """Multi-line status of all assigned behaviors."""
        if not self.behaviors:
            return "(no behaviors assigned — all ships idle)"
        lines = []
        for cfg in self.behaviors.values():
            line = f"  {cfg.ship_symbol}: {cfg.behavior_type} [{cfg.current_state}]"
            if cfg.error_message:
                line += f" — ERROR: {cfg.error_message}"
            lines.append(line)
        return "\n".join(lines)

    def tick(self, ship_symbol: str, fleet, client) -> Optional[str]:
        """
        Execute one step of a ship's behavior.
        Returns an alert string if LLM intervention is needed, else None.
        """
        config = self.behaviors.get(ship_symbol)
        if config is None:
            return None

        if config.behavior_type == BEHAVIOR_MINING_LOOP:
            return self._tick_mining_loop(config, fleet, client)

        if config.behavior_type == BEHAVIOR_SATELLITE_SCOUT:
            return self._tick_satellite_scout(config, fleet, client)

        return None

    def _tick_mining_loop(self, config: BehaviorConfig, fleet, client) -> Optional[str]:
        """
        Phase 1 static mining loop. Ships at an asteroid extract ore and
        jettison non-target cargo. Navigation is not handled yet.

        Parameters:
            asteroid_wp  — waypoint to mine at (ship must already be there)
            ore_types    — list of ore symbols to keep; all others are jettisoned.
                           If empty, keeps everything and alerts when cargo is full.
        """
        ship = fleet.get_ship(config.ship_symbol)
        if ship is None:
            return None  # Fleet not yet synced from API

        # If ship is on cooldown, nothing to do this tick
        if not ship.is_available():
            return None

        state = config.current_state
        ship_symbol = config.ship_symbol
        ore_types = config.parameters.get("ore_types", [])

        # ── START: ensure orbit ──────────────────────────────────────────
        if state == STATE_START:
            if ship.nav_status == "DOCKED":
                result = client.orbit(ship_symbol)
                if isinstance(result, dict) and "error" in result:
                    return self._error(config, f"failed to orbit: {result['error']}")
                ship.nav_status = "IN_ORBIT"
                fleet.persist()
            config.current_state = STATE_EXTRACT
            self._save()
            return None

        # ── EXTRACT ──────────────────────────────────────────────────────
        if state == STATE_EXTRACT:
            # Check cargo before extracting
            if ship.cargo_capacity > 0 and ship.cargo_units >= ship.cargo_capacity:
                config.current_state = STATE_CARGO_FULL
                self._save()
                return None

            result = client.extract(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                if "cooldown" in err.lower():
                    # Transient cooldown — wait and retry next tick
                    fleet.set_extraction_cooldown(ship_symbol, 60)
                    return None
                return self._error(config, f"extract failed: {err}")

            # Track cooldown from response
            cooldown_secs = result.get("cooldown", {}).get("remainingSeconds", 0)
            if cooldown_secs > 0:
                fleet.set_extraction_cooldown(ship_symbol, cooldown_secs)

            # Update cargo from response and jettison non-target ores
            cargo = result.get("cargo", {})
            inventory = cargo.get("inventory", [])
            ship.cargo_units = cargo.get("units", ship.cargo_units)

            if ore_types:
                for item in inventory:
                    if item["symbol"] not in ore_types:
                        client.jettison(ship_symbol, item["symbol"], item["units"])
                ship.cargo_units = sum(
                    i["units"] for i in inventory if i["symbol"] in ore_types
                )

            fleet.persist()
            self._save()
            return None

        # ── CARGO_FULL: alert until LLM intervenes ───────────────────────
        if state == STATE_CARGO_FULL:
            ore_desc = ", ".join(ore_types) if ore_types else "all ores"
            return (
                f"{ship_symbol} MINING_LOOP: cargo full ({ore_desc}). "
                f"Use cancel_behavior to sell manually, then reassign."
            )

        # ── ERROR: alert until LLM intervenes ───────────────────────────
        if state == STATE_ERROR:
            return f"{ship_symbol} MINING_LOOP suspended: {config.error_message}"

        return None

    def _tick_satellite_scout(self, config: BehaviorConfig, fleet, client) -> Optional[str]:
        """
        Satellite market scouting loop.

        Each satellite cycles through a shared market list starting at its own
        offset, so N satellites cover M markets with minimal overlap.

        Parameters:
            market_list    — ordered list of waypoint symbols to visit
            current_index  — current position in the list (advances each cycle)
        """
        ship = fleet.get_ship(config.ship_symbol)
        if ship is None or not ship.is_available():
            return None

        market_list = config.parameters.get("market_list", [])
        if not market_list:
            # No explicit list — use all known markets from cache (picks up new ones over time)
            from tools import load_market_cache
            market_list = sorted(load_market_cache().keys())
        if not market_list:
            # Cache still empty (startup race) — wait quietly until populated
            return None

        current_index = config.parameters.get("current_index", 0) % len(market_list)
        target_wp = market_list[current_index]
        # Derive system symbol from waypoint (e.g. "X1-AB12-A1" → "X1-AB12")
        system = "-".join(target_wp.split("-")[:2])
        state = config.current_state

        # ── START: scout current location if it has a market, then navigate ─
        if state == STATE_START:
            if ship.location:
                current_system = "-".join(ship.location.split("-")[:2])
                market_data = client.get_market(current_system, ship.location)
                if isinstance(market_data, dict) and "error" not in market_data:
                    from tools import _save_market_to_cache
                    _save_market_to_cache(ship.location, market_data)
            config.current_state = STATE_NAVIGATE
            self._save()
            return None

        # ── NAVIGATE: fly to target market ───────────────────────────────
        if state == STATE_NAVIGATE:
            # Ensure in orbit before navigating
            if ship.nav_status == "DOCKED":
                result = client.orbit(config.ship_symbol)
                if isinstance(result, dict) and "error" in result:
                    return self._error(config, f"failed to orbit: {result['error']}")
                ship.nav_status = "IN_ORBIT"
                fleet.persist()

            result = client.navigate(config.ship_symbol, target_wp)
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                # "already at destination" — skip navigation, go scout
                if "already" in err.lower() or "destination" in err.lower():
                    config.current_state = STATE_SCOUT
                    self._save()
                    return None
                # "not in orbit" — orbit and retry next tick
                if "orbit" in err.lower():
                    ship.nav_status = "DOCKED"  # force re-check next tick
                    fleet.persist()
                    return None
                return self._error(config, f"navigate failed: {err}")

            # Parse transit duration and mark ship as in-transit
            arrival_str = result.get("nav", {}).get("route", {}).get("arrival", "")
            if arrival_str:
                try:
                    arrival = datetime.fromisoformat(arrival_str.replace("Z", "+00:00"))
                    remaining = (arrival - datetime.now(timezone.utc)).total_seconds()
                    if remaining > 0:
                        fleet.set_transit(config.ship_symbol, remaining)
                except ValueError:
                    pass

            config.current_state = STATE_SCOUT
            self._save()
            return None

        # ── SCOUT: view market and save to cache ─────────────────────────
        if state == STATE_SCOUT:
            market_data = client.get_market(system, target_wp)
            if isinstance(market_data, dict) and "error" not in market_data:
                from tools import _save_market_to_cache
                _save_market_to_cache(target_wp, market_data)

            # Advance to next market and loop back to navigate
            config.parameters["current_index"] = (current_index + 1) % len(market_list)
            config.current_state = STATE_NAVIGATE
            self._save()
            return None

        # ── ERROR ────────────────────────────────────────────────────────
        if state == STATE_ERROR:
            return f"{config.ship_symbol} SATELLITE_SCOUT suspended: {config.error_message}"

        return None

    def _error(self, config: BehaviorConfig, message: str) -> str:
        config.current_state = STATE_ERROR
        config.error_message = message
        self._save()
        return f"{config.ship_symbol} MINING_LOOP error: {message}"
