import time
import logging
import math
from typing import List, Dict, Optional, Any
from threading import Thread, Lock
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

from ship_status import FleetTracker
from tools import get_engine, ALL_TOOLS, client, set_fleet, set_alert_queue, WAITING_TOOLS, STATE_CHANGING_TOOLS

# Setup basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

app = FastAPI(title="SpaceTraders Engine API")

# Initialize global state and inject into tools
fleet = FleetTracker()
set_fleet(fleet)

behavior_engine = get_engine()
alert_queue: List[str] = []
set_alert_queue(alert_queue)  # Give tools.py access to alerts
state_lock = Lock()  # Ensures thread-safe ticks & API calls
TICK_INTERVAL = 10  # seconds

# -------------------------
# Data models
# -------------------------
class BehaviorAssign(BaseModel):
    ship_symbol: str
    steps: str
    start_step: int = 0

class FleetAssign(BaseModel):
    ship_symbols: List[str]

# -------------------------
# Action Queue
# -------------------------
MAX_QUEUED_ACTIONS = 3

class ActionQueue:
    def __init__(self):
        self.queues = {}

    def enqueue(self, ship_symbol: str, tool_name: str, args: dict) -> str:
        q = self.queues.setdefault(ship_symbol, [])
        if len(q) >= MAX_QUEUED_ACTIONS:
            return f"Error: Queue full for {ship_symbol} ({MAX_QUEUED_ACTIONS} pending)."
        q.append((tool_name, args))
        return f"Queued: {tool_name} for {ship_symbol} (will execute after ship is available, {len(q)} in queue)"

    def get_ready(self, ship_symbol: str):
        q = self.queues.get(ship_symbol, [])
        return q.pop(0) if q else None

    def has_queued(self, ship_symbol: str) -> bool:
        return bool(self.queues.get(ship_symbol))

action_queue = ActionQueue()

# -------------------------
# Background Tick Loop
# -------------------------
def tick_loop():
    ticks = 0
    while True:
        with state_lock:
            behavior_engine.sync_state()

            # Sync all ships from API every ~60 seconds (ticks % 6 == 0, 6 * 10s = 60s)
            if ticks % 6 == 0:
                try:
                    ships_data = client.list_ships()
                    if isinstance(ships_data, list):
                        fleet.update_from_api(ships_data)
                except Exception as e:
                    log.error(f"Background fleet sync failed: {e}")
            ticks += 1

            # 1. Process Action Queue for available ships
            for ship_symbol in list(fleet.ships):
                ship_status = fleet.get_ship(ship_symbol)
                if ship_status and ship_status.is_available() and action_queue.has_queued(ship_symbol):
                    action = action_queue.get_ready(ship_symbol)
                    if action:
                        tool_name, tool_args = action
                        log.info(f"Executing queued {tool_name} for {ship_symbol}")
                        tool_func = next((t for t in ALL_TOOLS if t.name == tool_name), None)
                        if tool_func:
                            try:
                                tool_func.invoke(input=tool_args)
                            except Exception as e:
                                log.error(f"Queued {tool_name} failed: {e}")

            # 2. Process Behaviors
            for ship_symbol in list(fleet.ships):
                try:
                    alert = behavior_engine.tick(ship_symbol, fleet, client)
                    cfg = behavior_engine.behaviors.get(ship_symbol)
                    if cfg and cfg.last_action:
                        log.debug(f"[{ship_symbol}] {cfg.last_action}")
                    if alert and alert not in alert_queue:
                        alert_queue.append(alert)
                except Exception as e:
                    log.error(f"Error ticking {ship_symbol}: {e}")
        time.sleep(TICK_INTERVAL)

# Start tick loop in background
Thread(target=tick_loop, daemon=True).start()

# -------------------------
# Fleet / Behavior Endpoints
# -------------------------
@app.get("/api/state")
def get_state():
    with state_lock:
        return {
            "fleet": {
                s_name: {
                    "symbol": s.symbol,
                    "role": s.role,
                    "location": s.location,
                    "nav_status": s.nav_status,
                    "fuel_current": s.fuel_current,
                    "fuel_capacity": s.fuel_capacity,
                    "cargo_units": s.cargo_units,
                    "cargo_capacity": s.cargo_capacity,
                    "available_at": s.available_at,
                    "busy_reason": s.busy_reason,
                }
                for s_name, s in fleet.ships.items()
            },
            "behaviors": {
                k: {
                    "ship_symbol": v.ship_symbol,
                    "steps_str": v.steps_str,
                    "current_step_index": v.current_step_index,
                    "step_phase": v.step_phase,
                    "paused": v.paused,
                    "error_message": v.error_message,
                    "alert_sent": v.alert_sent,
                    "last_action": v.last_action,
                }
                for k, v in behavior_engine.behaviors.items()
            },
            "alerts": list(alert_queue),
        }

@app.post("/api/behaviors/assign")
def assign_behavior(payload: BehaviorAssign):
    with state_lock:
        try:
            result = behavior_engine.assign(
                payload.ship_symbol, payload.steps, start_step=payload.start_step
            )
            return {"status": "ok", "result": result}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/alerts")
def get_alerts():
    with state_lock:
        return list(alert_queue)

@app.delete("/api/alerts/{alert_index}")
def ack_alert(alert_index: int):
    with state_lock:
        try:
            removed = alert_queue.pop(alert_index)
            return {"status": "ok", "removed": removed}
        except IndexError:
            raise HTTPException(status_code=404, detail="Alert index out of range")

# -------------------------
# Formatted Game State Endpoint
# -------------------------
@app.get("/api/game_state")
def get_game_state():
    """Generates the formatted string context block for the LLM."""
    with state_lock:
        from bot import gather_game_state
        state_str = gather_game_state(fleet)
        state_str += f"\n\n[Behavior Status]\n{behavior_engine.summary()}"
        return {"state": state_str}

# -------------------------
# Automatic Tool Exposure
# -------------------------
def make_endpoint(tool_obj):
    def tool_endpoint(payload: dict = Body(default_factory=dict)):
        with state_lock:
            try:
                args = payload or {}
                ship_symbol = args.get("ship_symbol") or args.get("from_ship")

                # Handle Action Queuing
                if ship_symbol and tool_obj.name in (WAITING_TOOLS | STATE_CHANGING_TOOLS):
                    ship_status = fleet.get_ship(ship_symbol)
                    if ship_status and not ship_status.is_available():
                        has_behavior = ship_symbol in behavior_engine.behaviors
                        if has_behavior:
                            secs = ship_status.seconds_until_available()
                            raise Exception(f"Error: {ship_symbol} is busy ({ship_status.busy_reason}, {secs:.0f}s remaining) and has an active behavior. cancel_behavior first.")
                        elif tool_obj.name in STATE_CHANGING_TOOLS:
                            result = action_queue.enqueue(ship_symbol, tool_obj.name, args)
                            return {"result": result}
                        else:
                            secs = ship_status.seconds_until_available()
                            raise Exception(f"Error: {ship_symbol} is busy ({ship_status.busy_reason}, {secs:.0f}s remaining).")

                result = tool_obj.invoke(input=args)
                return {"result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    return tool_endpoint

# Expose every @tool function from your LangChain registry automatically
for t in ALL_TOOLS:
    endpoint = f"/api/tools/{t.name}"
    app.post(endpoint)(make_endpoint(t))
