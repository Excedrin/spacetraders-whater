"""
ship_status.py — Track ship cooldowns and availability.

Maintains local state about which ships are busy (on cooldown, in transit)
so the bot can make decisions about which ships to use without waiting.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ShipStatus:
    """Status of a single ship."""

    symbol: str
    role: str = "UNKNOWN"
    location: str = ""
    nav_status: str = "DOCKED"  # DOCKED, IN_ORBIT, IN_TRANSIT
    fuel_current: int = 0
    fuel_capacity: int = 0
    cargo_units: int = 0
    cargo_capacity: int = 0
    cargo_inventory: list = field(default_factory=list)  # List of {symbol, units}
    engine_speed: int = 30  # Ship engine speed for travel calculations

    # Cooldown/transit tracking
    available_at: float = 0.0  # Unix timestamp when ship becomes available
    busy_reason: str = ""  # "extraction_cooldown", "in_transit", etc.

    def is_available(self) -> bool:
        """Check if ship is available for commands."""
        return time.time() >= self.available_at

    def seconds_until_available(self) -> float:
        """Seconds until ship is available, or 0 if available now."""
        return max(0.0, self.available_at - time.time())

    def set_cooldown(self, seconds: float, reason: str):
        """Mark ship as busy for the given duration."""
        self.available_at = time.time() + seconds
        self.busy_reason = reason

    def clear_cooldown(self):
        """Mark ship as available."""
        self.available_at = 0.0
        self.busy_reason = ""

    def summary(self) -> str:
        """One-line summary of ship status."""
        status = f"{self.symbol} ({self.role})"
        status += f" @ {self.location}" if self.location else ""
        status += f" [{self.nav_status}]"
        status += f" Fuel:{self.fuel_current}/{self.fuel_capacity}"
        status += f" Cargo:{self.cargo_units}/{self.cargo_capacity}"

        if not self.is_available():
            secs = self.seconds_until_available()
            status += f" [BUSY: {self.busy_reason}, {secs:.0f}s]"

        return status


class FleetTracker:
    """
    Tracks status of all ships in the fleet.

    Syncs with API data and tracks local cooldown state.
    """

    def __init__(self):
        self.ships: dict[str, ShipStatus] = {}
        self._state_file = Path("fleet_state.json")
        self.last_persist = 0
        self._load()

    def _load(self):
        """Load persisted fleet state."""
        if self._state_file.exists():
            try:
                self.last_persist = os.stat(self._state_file).st_mtime
                data = json.loads(self._state_file.read_text())
                for ship_data in data.get("ships", []):
                    ship = ShipStatus(
                        symbol=ship_data["symbol"],
                        role=ship_data.get("role", "UNKNOWN"),
                        location=ship_data.get("location", ""),
                        nav_status=ship_data.get("nav_status", "DOCKED"),
                        fuel_current=ship_data.get("fuel_current", 0),
                        fuel_capacity=ship_data.get("fuel_capacity", 0),
                        cargo_units=ship_data.get("cargo_units", 0),
                        cargo_capacity=ship_data.get("cargo_capacity", 0),
                        cargo_inventory=ship_data.get("cargo_inventory", []),
                        engine_speed=ship_data.get("engine_speed", 30),
                        available_at=ship_data.get("available_at", 0.0),
                        busy_reason=ship_data.get("busy_reason", ""),
                    )
                    self.ships[ship.symbol] = ship
            except (json.JSONDecodeError, KeyError):
                pass

    def _check_reload(self):
        stat_result = os.stat(self._state_file)
        if self.last_persist < stat_result.st_mtime:
            self._load()

    def persist(self):
        """Save fleet state."""
        data = {
            "ships": [
                {
                    "symbol": s.symbol,
                    "role": s.role,
                    "location": s.location,
                    "nav_status": s.nav_status,
                    "fuel_current": s.fuel_current,
                    "fuel_capacity": s.fuel_capacity,
                    "cargo_units": s.cargo_units,
                    "cargo_capacity": s.cargo_capacity,
                    "cargo_inventory": s.cargo_inventory,
                    "engine_speed": s.engine_speed,
                    "available_at": s.available_at,
                    "busy_reason": s.busy_reason,
                }
                for s in self.ships.values()
            ]
        }
        self._state_file.write_text(json.dumps(data, indent=2))
        self.last_persist = time.time()

    def update_from_api(self, ships_data: list[dict]):
        """Update fleet status from API response (list_ships)."""
        for ship_data in ships_data:
            symbol = ship_data.get("symbol", "")
            if not symbol:
                continue

            nav = ship_data.get("nav", {})
            fuel = ship_data.get("fuel", {})
            cargo = ship_data.get("cargo", {})
            reg = ship_data.get("registration", {})
            engine = ship_data.get("engine", {})

            if symbol in self.ships:
                ship = self.ships[symbol]
            else:
                ship = ShipStatus(symbol=symbol)
                self.ships[symbol] = ship

            ship.role = reg.get("role", "UNKNOWN")
            ship.location = nav.get("waypointSymbol", "")
            ship.nav_status = nav.get("status", "DOCKED")
            ship.fuel_current = fuel.get("current", 0)
            ship.fuel_capacity = fuel.get("capacity", 0)
            ship.cargo_units = cargo.get("units", 0)
            ship.cargo_capacity = cargo.get("capacity", 0)
            ship.cargo_inventory = cargo.get("inventory", [])
            ship.engine_speed = engine.get("speed", 30)

            # Check if ship is in transit
            if ship.nav_status == "IN_TRANSIT":
                route = nav.get("route", {})
                arrival_str = route.get("arrival")
                if arrival_str:
                    try:
                        arrival = datetime.fromisoformat(
                            arrival_str.replace("Z", "+00:00")
                        )
                        remaining = (
                            arrival - datetime.now(timezone.utc)
                        ).total_seconds()
                        if remaining > 0:
                            ship.set_cooldown(remaining, "in_transit")
                    except ValueError:
                        pass

        self.persist()

    def update_ship_partial(self, symbol: str, data: dict):
        """Intercepts action payloads to keep local state perfectly synced.

        Pipes data from dock, orbit, navigate, buy, sell, extract, etc. directly
        into the tracker without needing GET /my/ships/{symbol} calls.
        """
        self._check_reload()
        if symbol not in self.ships:
            return  # Skip if we don't track this ship yet

        ship = self.ships[symbol]

        if "nav" in data:
            ship.location = data["nav"].get("waypointSymbol", ship.location)
            ship.nav_status = data["nav"].get("status", ship.nav_status)
        if "fuel" in data:
            ship.fuel_current = data["fuel"].get("current", ship.fuel_current)
            ship.fuel_capacity = data["fuel"].get("capacity", ship.fuel_capacity)
        if "cargo" in data:
            ship.cargo_units = data["cargo"].get("units", ship.cargo_units)
            ship.cargo_capacity = data["cargo"].get("capacity", ship.cargo_capacity)
            if "inventory" in data["cargo"]:
                ship.cargo_inventory = data["cargo"]["inventory"]
        if "cooldown" in data:
            cd = data["cooldown"].get("remainingSeconds", 0)
            if cd > 0:
                ship.set_cooldown(cd, "api_cooldown")

        self.persist()

    def set_extraction_cooldown(self, ship_symbol: str, seconds: float):
        """Mark a ship as on extraction cooldown."""
        self._check_reload()
        if ship_symbol in self.ships:
            self.ships[ship_symbol].set_cooldown(seconds, "extraction_cooldown")
            self.persist()

    def set_transit(self, ship_symbol: str, seconds: float):
        """Mark a ship as in transit."""
        self._check_reload()
        if ship_symbol in self.ships:
            self.ships[ship_symbol].set_cooldown(seconds, "in_transit")
            self.ships[ship_symbol].nav_status = "IN_TRANSIT"
            self.persist()

    def mark_available(self, ship_symbol: str):
        """Mark a ship as available."""
        self._check_reload()
        if ship_symbol in self.ships:
            self.ships[ship_symbol].clear_cooldown()
            self.persist()

    def get_available_ships(self) -> list[ShipStatus]:
        """Get list of ships that are currently available."""
        self._check_reload()
        return [s for s in self.ships.values() if s.is_available()]

    def get_busy_ships(self) -> list[ShipStatus]:
        """Get list of ships that are currently busy."""
        self._check_reload()
        return [s for s in self.ships.values() if not s.is_available()]

    def get_ship(self, symbol: str) -> Optional[ShipStatus]:
        """Get status for a specific ship."""
        self._check_reload()
        return self.ships.get(symbol)

    def fleet_summary(self) -> str:
        """Get a summary of the entire fleet for narrative context."""
        self._check_reload()
        if not self.ships:
            return "(No ships tracked yet)"

        lines = []
        for ship in sorted(self.ships.values(), key=lambda s: s.symbol):
            lines.append(f"• {ship.summary()}")
        return "\n".join(lines)

    def available_summary(self) -> str:
        """Get summary of available ships only."""
        self._check_reload()
        available = self.get_available_ships()
        if not available:
            busy = self.get_busy_ships()
            if busy:
                # Report when next ship will be available
                next_available = min(busy, key=lambda s: s.available_at)
                secs = next_available.seconds_until_available()
                return f"(All ships busy. {next_available.symbol} available in {secs:.0f}s)"
            return "(No ships)"

        lines = []
        for ship in sorted(available, key=lambda s: s.symbol):
            lines.append(f"• {ship.summary()}")
        return "\n".join(lines)
