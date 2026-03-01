"""
behaviors.py — Step-sequence ship behavior engine.

Ships run automated step sequences defined as comma-separated action lists.
The LLM defines behaviors like:
  "mine X1-AST IRON_ORE, goto X1-MKT, sell IRON_ORE, goto X1-AST, repeat"

Each step auto-handles dock/orbit transitions internally.
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

BEHAVIORS_FILE = Path("behaviors.json")


class StepType(Enum):
    MINE = "mine"        # mine WAYPOINT [ORE1 ORE2...] — extract until full
    GOTO = "goto"        # goto WAYPOINT — navigate, wait for arrival
    BUY  = "buy"         # buy ITEM
    SELL = "sell"        # sell ITEM or sell * — auto-dock, sell
    DELIVER = "deliver"  # deliver CONTRACT ITEM [UNITS] — auto-dock
    REFUEL = "refuel"    # refuel — auto-dock, buy fuel
    SCOUT = "scout"      # scout — view market, save to cache
    ALERT = "alert"      # alert MESSAGE — pause, notify LLM
    REPEAT = "repeat"    # repeat — restart from step 1
    NEGOTIATE = "negotiate" # negotiate — auto-dock, get contract
    BUY_SHIP = "buy_ship"   #  SHIP_TYPE — auto-dock, buy ship


@dataclass
class Step:
    step_type: StepType
    args: list[str] = field(default_factory=list)

    def __str__(self):
        if self.args:
            return f"{self.step_type.value} {' '.join(self.args)}"
        return self.step_type.value


def parse_steps(steps_str: str) -> list[Step]:
    """Parse a comma-separated step string into Step objects.

    Examples:
        "mine X1-AST IRON_ORE, goto X1-MKT, sell IRON_ORE, repeat"
        "goto X1-MKT, scout, goto X1-MKT2, scout, repeat"
    """
    steps = []
    for part in steps_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        verb = tokens[0].lower()
        args = tokens[1:]
        try:
            step_type = StepType(verb)
        except ValueError:
            raise ValueError(f"Unknown step type: '{verb}'. Valid: {[s.value for s in StepType]}")
        steps.append(Step(step_type=step_type, args=args))
    return steps


@dataclass
class BehaviorConfig:
    ship_symbol: str
    steps: list[Step]
    steps_str: str               # original string for display + repeat
    current_step_index: int = 0
    step_phase: str = "INIT"     # sub-phase within a step
    paused: bool = False         # True when alert fired, waiting for LLM
    error_message: str = ""
    alert_sent: bool = False


_engine_instance: Optional["BehaviorEngine"] = None


def get_engine() -> "BehaviorEngine":
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BehaviorEngine()
    return _engine_instance


class BehaviorEngine:
    """
    Manages autonomous ship behaviors as step sequences.

    Call tick() for each ship every loop. It advances the ship's current step
    one phase, returning an alert string when LLM intervention is required,
    or None if everything is running smoothly.
    """

    def __init__(self):
        self.behaviors: dict[str, BehaviorConfig] = {}
        self._last_mtime = 0.0  # Track file timestamp
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        if not BEHAVIORS_FILE.exists():
            return

        # Update timestamp tracking
        try:
            current_mtime = BEHAVIORS_FILE.stat().st_mtime
            self._last_mtime = current_mtime
        except OSError:
            pass

        try:
            entries = json.loads(BEHAVIORS_FILE.read_text())
            for e in entries:
                # New format: has steps_str
                if "steps_str" in e:
                    try:
                        steps = parse_steps(e["steps_str"])
                    except ValueError:
                        continue
                    cfg = BehaviorConfig(
                        ship_symbol=e["ship_symbol"],
                        steps=steps,
                        steps_str=e["steps_str"],
                        current_step_index=e.get("current_step_index", 0),
                        step_phase=e.get("step_phase", "INIT"),
                        paused=e.get("paused", False),
                        error_message=e.get("error_message", ""),
                        alert_sent=e.get("alert_sent", False),
                    )
                    self.behaviors[cfg.ship_symbol] = cfg

                # Backward compat: old MINING_LOOP format
                elif e.get("behavior_type") == "MINING_LOOP":
                    params = e.get("parameters", {})
                    asteroid = params.get("asteroid_wp", "UNKNOWN")
                    ores = params.get("ore_types", [])
                    ore_str = " ".join(ores) if ores else ""
                    mine_part = f"mine {asteroid} {ore_str}".strip()
                    steps_str = f"{mine_part}, alert cargo full, repeat"
                    try:
                        steps = parse_steps(steps_str)
                    except ValueError:
                        continue
                    cfg = BehaviorConfig(
                        ship_symbol=e["ship_symbol"],
                        steps=steps,
                        steps_str=steps_str,
                    )
                    self.behaviors[cfg.ship_symbol] = cfg

                # Backward compat: old SATELLITE_SCOUT format
                elif e.get("behavior_type") == "SATELLITE_SCOUT":
                    params = e.get("parameters", {})
                    markets = params.get("market_list", [])
                    if markets:
                        parts = []
                        for m in markets:
                            parts.append(f"goto {m}")
                            parts.append("scout")
                        parts.append("repeat")
                        steps_str = ", ".join(parts)
                    else:
                        steps_str = "alert no markets configured"
                    try:
                        steps = parse_steps(steps_str)
                    except ValueError:
                        continue
                    cfg = BehaviorConfig(
                        ship_symbol=e["ship_symbol"],
                        steps=steps,
                        steps_str=steps_str,
                    )
                    self.behaviors[cfg.ship_symbol] = cfg

        except Exception as exc:
            log.warning("Failed to load behaviors.json: %s", exc)

    def _save(self):
        BEHAVIORS_FILE.write_text(json.dumps(
            [
                {
                    "ship_symbol": c.ship_symbol,
                    "steps_str": c.steps_str,
                    "current_step_index": c.current_step_index,
                    "step_phase": c.step_phase,
                    "paused": c.paused,
                    "error_message": c.error_message,
                    "alert_sent": c.alert_sent,
                }
                for c in self.behaviors.values()
            ],
            indent=2,
        ))

        # Update our local timestamp so we don't reload our own changes
        try:
            self._last_mtime = BEHAVIORS_FILE.stat().st_mtime
        except OSError:
            pass

    def sync_state(self):
        """Check if file has changed on disk (e.g. by CLI) and reload if necessary."""
        if not BEHAVIORS_FILE.exists():
            return

        try:
            disk_mtime = BEHAVIORS_FILE.stat().st_mtime
            # If disk file is newer than our last load/save, reload it
            if disk_mtime > self._last_mtime:
                log.info("Behaviors file changed on disk. Reloading...")
                self.behaviors.clear()
                self._load()
        except OSError:
            pass

    # ── Public API ───────────────────────────────────────────────────────

    def assign(self, ship_symbol: str, steps_str: str, start_step: int = 0) -> str:
        """Assign a step-sequence behavior to a ship. Returns confirmation or error.

        Args:
            start_step: Index of the step to start at (0-based). Useful for spreading
                        multiple ships across the same sequence so they don't all pile
                        up at the first step.
        """
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
        suffix = f" (starting at step {clamped + 1})" if clamped else ""
        return f"Assigned behavior to {ship_symbol}{suffix}: {steps_str}"

    def cancel(self, ship_symbol: str) -> None:
        """Remove a ship's behavior. It returns to IDLE (LLM control)."""
        self.behaviors.pop(ship_symbol, None)
        self._save()

    def resume(self, ship_symbol: str) -> str:
        """Resume a paused behavior (after an alert was handled)."""
        cfg = self.behaviors.get(ship_symbol)
        if cfg is None:
            return f"{ship_symbol} has no behavior to resume."
        if not cfg.paused:
            return f"{ship_symbol} is not paused."
        cfg.paused = False
        cfg.alert_sent = False
        # Advance past the alert step
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        if cfg.current_step_index >= len(cfg.steps):
            cfg.current_step_index = 0
        self._save()
        step = cfg.steps[cfg.current_step_index]
        return f"Resumed {ship_symbol}. Now on step {cfg.current_step_index + 1}/{len(cfg.steps)}: {step}"

    def skip_step(self, ship_symbol: str) -> str:
        """Skip the current step and advance to the next."""
        cfg = self.behaviors.get(ship_symbol)
        if cfg is None:
            return f"{ship_symbol} has no behavior."
        old_step = cfg.steps[cfg.current_step_index]
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        cfg.paused = False
        cfg.alert_sent = False
        cfg.error_message = ""
        if cfg.current_step_index >= len(cfg.steps):
            cfg.current_step_index = 0
        self._save()
        new_step = cfg.steps[cfg.current_step_index]
        return f"Skipped '{old_step}' for {ship_symbol}. Now on step {cfg.current_step_index + 1}/{len(cfg.steps)}: {new_step}"

    def get_idle_ships(self, fleet) -> list[str]:
        """Ships with no assigned behavior — need LLM attention."""
        return [s for s in fleet.ships if s not in self.behaviors]

    def summary(self) -> str:
        """Multi-line status of all assigned behaviors."""
        if not self.behaviors:
            return "(no behaviors assigned -- all ships idle)"
        lines = []
        for cfg in self.behaviors.values():
            step_idx = cfg.current_step_index
            total = len(cfg.steps)
            current_step = cfg.steps[step_idx] if step_idx < total else "?"
            status = cfg.step_phase
            if cfg.paused:
                status = "PAUSED"
            if cfg.error_message:
                status = f"ERROR: {cfg.error_message}"

            lines.append(
                f"  {cfg.ship_symbol}: step {step_idx + 1}/{total} "
                f"[{current_step}] ({status})"
            )
            lines.append(f"    sequence: {cfg.steps_str}")
        return "\n".join(lines)

    # ── Tick: dispatch to step handler ───────────────────────────────────

    def tick(self, ship_symbol: str, fleet, client) -> Optional[str]:
        """Execute one phase of a ship's current step.
        Returns an alert string if LLM intervention is needed, else None.
        """
        cfg = self.behaviors.get(ship_symbol)
        if cfg is None:
            return None
        if cfg.paused:
            return None  # Waiting for LLM to resume/skip/cancel

        ship = fleet.get_ship(ship_symbol)
        if ship is None:
            return None  # Fleet not yet synced
        if not ship.is_available():
            return None  # Ship is on cooldown / in transit

        if cfg.current_step_index >= len(cfg.steps):
            cfg.current_step_index = 0
            cfg.step_phase = "INIT"

        step = cfg.steps[cfg.current_step_index]

        handlers = {
            StepType.MINE: self._step_mine,
            StepType.GOTO: self._step_goto,
            StepType.SELL: self._step_sell,
            StepType.DELIVER: self._step_deliver,
            StepType.REFUEL: self._step_refuel,
            StepType.SCOUT: self._step_scout,
            StepType.ALERT: self._step_alert,
            StepType.REPEAT: self._step_repeat,
            StepType.NEGOTIATE: self._step_negotiate,
            StepType.BUY_SHIP: self._step_buy_ship,
        }

        handler = handlers.get(step.step_type)
        if handler is None:
            return self._error(cfg, f"no handler for step type: {step.step_type}")

        try:
            return handler(cfg, step, ship, fleet, client)
        except Exception as exc:
            return self._error(cfg, f"exception in {step.step_type.value}: {exc}")

    # ── Step handlers ────────────────────────────────────────────────────

    def _step_mine(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """mine WAYPOINT [ORE1 ORE2...]
        INIT: navigate to asteroid if needed, orbit
        EXTRACTING: extract, jettison non-targets, loop until cargo full -> advance
        """
        ship_symbol = cfg.ship_symbol
        asteroid_wp = step.args[0] if step.args else None
        ore_types = step.args[1:] if len(step.args) > 1 else []

        if cfg.step_phase == "INIT":
            # Navigate to asteroid if not already there
            if asteroid_wp and ship.location != asteroid_wp:
                return self._auto_navigate(cfg, ship, fleet, client, asteroid_wp, next_phase="EXTRACTING")

            # Ensure orbit
            if ship.nav_status == "DOCKED":
                result = client.orbit(ship_symbol)
                if isinstance(result, dict) and "error" in result:
                    return self._error(cfg, f"orbit failed: {result['error']}")
                ship.nav_status = "IN_ORBIT"
                fleet.persist()

            cfg.step_phase = "EXTRACTING"
            self._save()
            return None

        if cfg.step_phase == "EXTRACTING":
            # Check cargo before extracting
            if ship.cargo_capacity > 0 and ship.cargo_units >= ship.cargo_capacity:
                self._advance(cfg)
                return None

            # Ensure orbit (may have docked for refuel)
            if ship.nav_status == "DOCKED":
                result = client.orbit(ship_symbol)
                if isinstance(result, dict) and "error" in result:
                    return self._error(cfg, f"orbit failed: {result['error']}")
                ship.nav_status = "IN_ORBIT"
                fleet.persist()

            result = client.extract(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                err = result["error"]
                if "cooldown" in err.lower():
                    fleet.set_extraction_cooldown(ship_symbol, 60)
                    return None
                return self._error(cfg, f"extract failed: {err}")

            # Track cooldown
            cooldown_secs = result.get("cooldown", {}).get("remainingSeconds", 0)
            if cooldown_secs > 0:
                fleet.set_extraction_cooldown(ship_symbol, cooldown_secs)

            # Update cargo and jettison non-targets
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

        # Unknown phase — reset
        cfg.step_phase = "INIT"
        self._save()
        return None

    def _step_buy(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """buy ITEM [UNITS] — auto-dock, purchase cargo."""
        if not step.args:
            return self._error(cfg, "buy requires ITEM argument")

        ship_symbol = cfg.ship_symbol
        trade_symbol = step.args[0]

        # Calculate demand
        # If arg provided (e.g. "buy FUEL 10"), use it. Otherwise fill cargo.
        requested = int(step.args[1]) if len(step.args) > 1 else 999999

        # 1. Ensure Docked
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock for buy failed: {result['error']}")
            ship.nav_status = "DOCKED"
            fleet.persist()

        # 2. Check Capacity
        # We need fresh cargo stats to be safe
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" not in cargo_data:
            ship.cargo_units = cargo_data.get("units", 0)
            ship.cargo_capacity = cargo_data.get("capacity", 0)

        available_space = ship.cargo_capacity - ship.cargo_units
        if available_space <= 0:
            # Cargo full? Advance. (Assumption: We already have what we need or can't buy more)
            self._advance(cfg)
            return None

        units_to_buy = min(requested, available_space)
        if units_to_buy <= 0:
            self._advance(cfg)
            return None

        # 3. Execute Purchase
        result = client.purchase_cargo(ship_symbol, trade_symbol, units_to_buy)

        if isinstance(result, dict) and "error" in result:
            err = result['error']
            # If insufficient funds, this is a critical strategy failure -> ALERT
            if "credits" in str(err).lower() or "funds" in str(err).lower():
                 return self._error(cfg, f"Insufficient credits to buy {trade_symbol}: {err}")
            return self._error(cfg, f"buy {trade_symbol} failed: {err}")

        # 4. Success - Update State & Advance
        tx = result.get("transaction", {})
        cargo = result.get("cargo", {})
        ship.cargo_units = cargo.get("units", ship.cargo_units)
        fleet.persist()

        self._advance(cfg)
        # Optional: return a log string if you want specific narrative logs for trades
        return None


    def _step_goto(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """goto WAYPOINT — navigate, wait for arrival."""
        if not step.args:
            return self._error(cfg, "goto requires a WAYPOINT argument")

        dest_wp = step.args[0]

        if cfg.step_phase == "INIT":
            if ship.location == dest_wp:
                # Already there
                self._advance(cfg)
                return None
            return self._auto_navigate(cfg, ship, fleet, client, dest_wp, next_phase="WAITING")

        if cfg.step_phase == "WAITING":
            # Ship arrived (is_available check at top of tick ensures this)
            if ship.location == dest_wp or ship.nav_status != "IN_TRANSIT":
                self._advance(cfg)
                return None
            # Still in transit — wait
            return None

        cfg.step_phase = "INIT"
        self._save()
        return None

    def _step_sell(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """sell ITEM or sell * — auto-dock, sell."""
        ship_symbol = cfg.ship_symbol
        target = step.args[0] if step.args else "*"

        # Auto-dock
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock failed: {result['error']}")
            ship.nav_status = "DOCKED"
            fleet.persist()

        # Get cargo
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" in cargo_data:
            return self._error(cfg, f"cargo check failed: {cargo_data['error']}")

        inventory = cargo_data.get("inventory", [])

        # Check for contract goods to protect
        from tools import _get_contract_goods
        contract_goods = _get_contract_goods()

        # Determine what to sell
        sold_any = False
        errors = []
        for item in inventory:
            item_sym = item["symbol"]
            item_units = item["units"]
            if item_units <= 0:
                continue

            # Skip contract goods
            if item_sym in contract_goods:
                continue

            # Filter by target
            if target != "*" and item_sym != target:
                continue

            result = client.sell_cargo(ship_symbol, item_sym, item_units)
            if isinstance(result, dict) and "error" in result:
                errors.append(f"{item_sym}: {result['error']}")
            else:
                sold_any = True

        # Update cargo after selling
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" not in cargo_data:
            ship.cargo_units = cargo_data.get("units", 0)
            fleet.persist()

        if errors and not sold_any:
            return self._error(cfg, f"sell failed: {'; '.join(errors)}")

        self._advance(cfg)
        return None

    def _step_deliver(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """deliver CONTRACT ITEM [UNITS] — auto-dock, deliver contract goods."""
        if len(step.args) < 2:
            return self._error(cfg, "deliver requires CONTRACT and ITEM arguments")

        ship_symbol = cfg.ship_symbol
        contract_id = step.args[0]
        trade_symbol = step.args[1]
        units = int(step.args[2]) if len(step.args) > 2 else None

        # Auto-dock
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock failed: {result['error']}")
            ship.nav_status = "DOCKED"
            fleet.persist()

        # Get available units
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" in cargo_data:
            return self._error(cfg, f"cargo check failed: {cargo_data['error']}")

        available = 0
        for item in cargo_data.get("inventory", []):
            if item["symbol"] == trade_symbol:
                available = item["units"]
                break

        if available <= 0:
            # No goods to deliver — advance anyway (may have already delivered)
            self._advance(cfg)
            return None

        deliver_units = min(units, available) if units else available

        result = client.deliver_contract(contract_id, ship_symbol, trade_symbol, deliver_units)
        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"deliver failed: {result['error']}")

        # Update cargo
        cargo_data = client.get_cargo(ship_symbol)
        if isinstance(cargo_data, dict) and "error" not in cargo_data:
            ship.cargo_units = cargo_data.get("units", 0)
            fleet.persist()

        self._advance(cfg)
        return None

    def _step_refuel(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """refuel — auto-dock, buy fuel."""
        ship_symbol = cfg.ship_symbol

        # Solar ships don't need fuel
        if ship.fuel_capacity == 0:
            self._advance(cfg)
            return None

        # Auto-dock
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock failed: {result['error']}")
            ship.nav_status = "DOCKED"
            fleet.persist()

        result = client.refuel(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"refuel failed: {result['error']}")

        fuel = result.get("fuel", {})
        ship.fuel_current = fuel.get("current", ship.fuel_current)
        fleet.persist()

        self._advance(cfg)
        return None

    def _step_scout(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """scout — auto-orbit, view market, save to cache."""
        ship_symbol = cfg.ship_symbol

        # Need to be at a waypoint (not in transit)
        if not ship.location:
            return self._error(cfg, "ship has no location")

        system = "-".join(ship.location.split("-")[:2])
        market_data = client.get_market(system, ship.location)
        if isinstance(market_data, dict) and "error" not in market_data:
            from tools import _save_market_to_cache
            _save_market_to_cache(ship.location, market_data)

        self._advance(cfg)
        return None

    def _step_negotiate(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """negotiate — auto-dock, negotiate new contract, alert."""
        ship_symbol = cfg.ship_symbol

        # 1. Ensure Docked
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock for negotiation failed: {result['error']}")
            ship.nav_status = "DOCKED"
            fleet.persist()

        # 2. Negotiate
        result = client.negotiate_contract(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"negotiation failed: {result['error']}")

        # 3. Alert the Human/LLM
        contract = result.get("contract", {})
        c_id = contract.get("id", "?")
        payment = contract.get("terms", {}).get("payment", {}).get("onAccepted", 0)

        # We pause here so the LLM can review the contract terms in the next turn
        cfg.paused = True
        cfg.alert_sent = True
        self._save()

        return f"{ship_symbol} NEGOTIATED contract {c_id} (upfront: {payment}). Review terms in [Contracts] and accept if profitable."

    def _step_buy_ship(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """buy_ship SHIP_TYPE — auto-dock, buy ship, alert."""
        if not step.args:
            return self._error(cfg, "buy_ship requires SHIP_TYPE argument")

        ship_type = step.args[0]
        ship_symbol = cfg.ship_symbol

        # 1. Ensure Docked
        if ship.nav_status != "DOCKED":
            client.dock(ship_symbol)
            ship.nav_status = "DOCKED"
            fleet.persist()

        # 2. buy_ship
        # Note: buy_ship_ship endpoint requires the *current waypoint*, not the ship symbol
        wp = ship.location
        result = client.buy_ship_ship(ship_type, wp)

        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"buy_ship {ship_type} failed: {result['error']}")

        # 3. Advance (Success)
        new_ship = result.get("ship", {})
        self._advance(cfg)
        return f"{ship_symbol} Bought new ship {new_ship.get('symbol')} ({ship_type})!"


    def _step_alert(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """alert MESSAGE — pause and notify LLM."""
        if not cfg.alert_sent:
            cfg.paused = True
            cfg.alert_sent = True
            self._save()
            message = " ".join(step.args) if step.args else "behavior paused"
            return f"{cfg.ship_symbol} ALERT: {message} (step {cfg.current_step_index + 1}/{len(cfg.steps)} of: {cfg.steps_str})"
        return None

    def _step_repeat(self, cfg, step, ship, fleet, client) -> Optional[str]:
        """repeat — restart from step 1."""
        cfg.current_step_index = 0
        cfg.step_phase = "INIT"
        self._save()
        return None

    # ── Helpers ──────────────────────────────────────────────────────────

    def _auto_navigate(self, cfg, ship, fleet, client, dest_wp, next_phase="WAITING") -> Optional[str]:
        """Navigate ship to dest_wp, handling orbit, auto-refuel, and transit tracking."""
        ship_symbol = cfg.ship_symbol

        # Ensure orbit before navigating
        if ship.nav_status == "DOCKED":
            result = client.orbit(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"orbit failed: {result['error']}")
            ship.nav_status = "IN_ORBIT"
            fleet.persist()

        result = client.navigate(ship_symbol, dest_wp)
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            # Already at destination
            if "already" in err.lower() or "destination" in err.lower():
                self._advance(cfg)
                return None
            # Insufficient fuel — try auto-refuel
            if "fuel" in err.lower() or "insufficient" in err.lower():
                refuel_result = self._try_auto_refuel(cfg, ship, fleet, client)
                if refuel_result is not None:
                    return refuel_result
                # Retry navigation after refueling
                result = client.navigate(ship_symbol, dest_wp)
                if isinstance(result, dict) and "error" in result:
                    return self._error(cfg, f"navigate failed after refuel: {result['error']}")
            else:
                return self._error(cfg, f"navigate to {dest_wp} failed: {err}")

        # Track transit duration
        arrival_str = result.get("nav", {}).get("route", {}).get("arrival", "")
        if arrival_str:
            try:
                arrival = datetime.fromisoformat(arrival_str.replace("Z", "+00:00"))
                remaining = (arrival - datetime.now(timezone.utc)).total_seconds()
                if remaining > 0:
                    fleet.set_transit(ship_symbol, remaining)
                    ship.location = dest_wp  # Will be there on arrival
            except ValueError:
                pass

        cfg.step_phase = next_phase
        self._save()
        return None

    def _try_auto_refuel(self, cfg, ship, fleet, client) -> Optional[str]:
        """Try to refuel at current location. Returns error string or None on success."""
        ship_symbol = cfg.ship_symbol

        # Check if current location has fuel
        from tools import load_market_cache
        market_cache = load_market_cache()
        current_market = market_cache.get(ship.location, {})
        has_fuel = "FUEL" in current_market.get("exchange", []) or "FUEL" in current_market.get("exports", [])

        if not has_fuel:
            return self._error(cfg, f"insufficient fuel at {ship.location} (no fuel market here)")

        # Dock and refuel
        if ship.nav_status != "DOCKED":
            result = client.dock(ship_symbol)
            if isinstance(result, dict) and "error" in result:
                return self._error(cfg, f"dock for refuel failed: {result['error']}")
            ship.nav_status = "DOCKED"

        result = client.refuel(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"refuel failed: {result['error']}")

        fuel = result.get("fuel", {})
        ship.fuel_current = fuel.get("current", ship.fuel_current)
        fleet.persist()

        # Re-orbit for navigation
        result = client.orbit(ship_symbol)
        if isinstance(result, dict) and "error" in result:
            return self._error(cfg, f"re-orbit after refuel failed: {result['error']}")
        ship.nav_status = "IN_ORBIT"
        fleet.persist()

        return None  # Success

    def _advance(self, cfg):
        """Advance to the next step."""
        cfg.current_step_index += 1
        cfg.step_phase = "INIT"
        cfg.error_message = ""
        cfg.alert_sent = False
        if cfg.current_step_index >= len(cfg.steps):
            # Behavior complete (no repeat step) — remove it
            self.behaviors.pop(cfg.ship_symbol, None)
        self._save()

    def _error(self, cfg, message: str) -> str:
        """Set error state and return alert string."""
        cfg.error_message = message
        cfg.paused = True
        cfg.alert_sent = True
        self._save()
        step_idx = cfg.current_step_index
        total = len(cfg.steps)
        current_step = cfg.steps[step_idx] if step_idx < total else "?"
        return (
            f"{cfg.ship_symbol} ERROR at step {step_idx + 1}/{total} [{current_step}]: "
            f"{message} (sequence: {cfg.steps_str})"
        )
