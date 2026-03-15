A distributed autonomous agent that separates the game engine (server with tick loop) from the strategic brain (LLM client). Ships execute step-sequence behaviors autonomously; the LLM intervenes only when alerts trigger or strategic review is needed.

### 🗺️ SpaceTraders Bot Code Map

#### 1. Architecture Overview
**Distributed Client-Server Model:**
- **Server (`server.py`):** FastAPI backend running a continuous tick loop that progresses ship behaviors, syncs fleet state, and manages an alert queue.
- **LLM Client (`bot.py`):** Lightweight agent that polls server state, reacts to alerts, and assigns new behaviors via HTTP endpoints.
- **Manual CLI (`play_cli.py`):** HTTP client for human-controlled tool invocation.

The core concept: **Fast Loop (Server Tick)** runs deterministic behaviors every ~10 seconds. **Slow Loop (LLM)** wakes on alerts or strategy timer (~10min) to make high-level decisions.

#### 2. Major Components (Files)

*   **`api_client.py` (The Communicator)**
    *   *Role:* Handles all HTTP requests to the official SpaceTraders API.
    *   *Key Features:* Thread-safe token bucket `RateLimiter` (2 req/sec), automatic pagination (`_paginate_request`), and basic GET caching.
    *   *Waypoint Methods:* `list_waypoints()` (paginated), `get_waypoint()` (single waypoint with traits/chart status).

*   **`ship_status.py` (The Fleet Tracker)**
    *   *Role:* Local state management for ships (`FleetTracker`, `ShipStatus`).
    *   *Key Features:* Tracks position, cargo (including `cargo_inventory`), fuel, cooldowns, engine speed. Persists to `fleet_state.json`.
    *   *Partial Updates:* `update_ship_partial()` intercepts API action responses to keep state perfectly synced without extra GET calls.

*   **`tools.py` (The Core Engine & Tool Registry)**
    *   *Role:* Game logic, behavior engine, and LangChain tool wrappers.
    *   *Layer 1: Core Logic:* `_navigate_ship_logic`, `_buy_cargo_logic`, `_extract_ore_logic`, etc. Handle math, pathfinding, API calls.
    *   *Layer 2: `BehaviorEngine`:* State machine executing step sequences (e.g., `goto -> mine -> goto -> sell -> autotrade`). Each step has phases (INIT, WAITING, etc.).
    *   *Layer 3: LangChain Tools:* `@tool`-decorated functions (e.g., `create_behavior`, `navigate_ship`) for LLM/CLI invocation.
    *   *Waypoint Cache:* `get_system_waypoints()` fetches and caches system waypoints with trait/type filtering. Metadata stored in `waypoint_cache.json` with `_systems_fetched` tracking.
    *   *Step Types:* `goto`, `mine`, `buy`, `sell`, `transfer_cargo`, `deliver_contract`, `refuel`, `dock`, `orbit`, `scout`, `chart`, `construct`, `explore`, `autotrade`, `supply`, `alert`, `repeat`, `stop`.

*   **`server.py` (The Game Engine Server)**
    *   *Role:* FastAPI server running the autonomous backend. Two major operations:
        1. **Tick Loop:** Background thread calling `BehaviorEngine.tick(ship, fleet, client)` every ~10 seconds. Processes behaviors and syncs fleet state from API.
        2. **REST Endpoints:** Exposes game state and tools over HTTP.
    *   *Endpoints:*
        - `GET /api/state` — Returns formatted game context for LLM decision-making.
        - `GET /api/game_state` — Full game state (ships, behaviors, alerts).
        - `POST /api/tools/{tool_name}` — Execute tools with smart action queuing (max 3 pending per ship).
        - `GET /api/alerts` — List behavior-generated alerts.
        - `DELETE /api/alerts/{index}` — Remove alert (for LLM to clear processed ones).
        - `POST /api/behaviors/assign` — Assign new behavior to ship.
    *   *State:* Manages `fleet_state.json`, `behaviors.json`, action queue.

*   **`bot.py` (The LLM Client)**
    *   *Role:* Lightweight agent loop that makes strategic decisions.
    *   *Key Logic:*
        1. Polls `GET /api/state` to detect idle ships or alerts.
        2. If idle or alert, fetches `/api/game_state` and calls Ollama for a decision.
        3. Executes chosen tool via `POST /api/tools/{tool_name}`.
        4. Clears processed alerts.
    *   *Tick Interval:* 10 seconds (matched to server tick for coherent timing).

*   **`narrative.py` & `events.py` (The Memory/Lore)**
    *   *Role:* Captain's log generation and event logging for context continuity.
    *   *State:* `story.jsonl` (append-only log), `events.jsonl` (tool execution log).

*   **`play_cli.py` (The Manual Override)**
    *   *Role:* Human-playable CLI that invokes tools via HTTP.
    *   *Communication:* Makes `requests.post()` calls to `http://localhost:8000/api/tools/{tool_name}`.

#### 3. How Things Fit Together (Data Flow)

**Server Tick (Fast Loop, ~every 10s):**
1. `server.py` background thread calls `behavior_engine.tick()` for each ship.
2. `BehaviorEngine.tick()` reads current step, dispatches to step handler (`_step_goto`, `_step_mine`, etc.).
3. Step handler calls core logic (e.g., `_navigate_ship_logic`) which makes API calls and updates local fleet tracker.
4. Fleet state persists to `fleet_state.json`. Action results push alerts to `alert_queue` if needed.

**LLM Wake-up (Slow Loop, ~every 10min or on alert):**
1. `bot.py` polls `GET /api/state` — checks for alerts or idle ships.
2. If work needed, fetches `GET /api/game_state` and calls Ollama with context.
3. LLM selects a tool (e.g., `create_behavior("WHATER-1", "goto X, buy Y, autotrade")`).
4. `bot.py` calls `POST /api/tools/create_behavior` with parameters.
5. `server.py` executes tool (updates `behaviors.json`, queues action if needed).
6. `bot.py` polls `GET /api/alerts` to see any new issues, clears processed alerts via DELETE.

#### 4. Key Functions / Classes to Note

**Server (`server.py`):**
- `app.backgroundtasks()`: Background thread calling `behavior_engine.tick()` in a loop.
- `@app.get("/api/state")`: Detects idle ships, formats behavioral status.
- `@app.post("/api/tools/{tool_name}")`: Routes tool execution with action queue logic.

**LLM Client (`bot.py`):**
- `run_agent()`: Main polling loop checking alerts and ship states.
- `_run_llm_cycle()`: Calls Ollama, processes tool responses, clears alerts.
- `gather_game_state()`: Builds formatted prompt from fleet + market data.

**Core Engine (`tools.py`):**
- `BehaviorEngine.tick()`: Main state machine dispatcher. Core of autonomous execution.
- `BehaviorEngine._step_*()`: Individual step handlers (goto, mine, buy, sell, autotrade, explore, etc.).
- `_navigate_ship_logic()`: Most complex core function (refueling, multi-hop pathfinding, mode selection).
- `_analyze_trade_routes()`: Scans market cache for profitable arbitrage, fleet-aware filtering.
- `get_system_waypoints()`: Fetches/caches waypoints with trait filtering (market, shipyard, charted).
- `FleetTracker.update_ship_partial()`: Intercepts API responses to keep local state in sync.


---

#### 5. Recent Updates & New Features (Implemented)

**New Behaviors & Steps (v1.3+):**
- `explore` — Automatically chart uncharted waypoints, scout markets and shipyards in a system.
- `construct` — Participate in jump gate construction projects.
- `chart` — Chart a specific uncharted waypoint.
- `supply` — Deliver goods for supply contracts.
- `autotrade` — Now **fleet-aware**: filters out goods being traded by other ships to avoid conflicts.

**Waypoint Caching Improvements:**
- `get_system_waypoints(system_symbol, waypoint_type=None, trait=None)` — Intelligent caching with metadata.
- Trait/type filtering (e.g., find all MARKETPLACE waypoints, UNCHARTED waypoints).
- `_systems_fetched` metadata tracks which systems have been fully cached (no pagination needed on re-fetch).

**Fleet State Tracking:**
- `ship.cargo_inventory` — Local tracking of cargo items (avoids GET /cargo calls).
- `ship.engine_speed` — Ship engine speed for pathfinding calculations.
- `FleetTracker.update_ship_partial()` — Intercepts tool response payloads to update state in-place without extra API calls.

**Step Sequence Enhancements:**
- `repeat [N]` syntax — Execute steps N times, or forever if N omitted.
- Better phase management (INIT, WAITING, ERROR states) for complex multi-step behaviors.

**Alert System:**
- Behaviors can raise alerts to wake the LLM (e.g., "cargo full, unable to find buyer").
- LLM can acknowledge/clear alerts via `DELETE /api/alerts/{index}`.

This design successfully separates the "Game Engine" (autonomous server tick loop) from the "Brain" (occasional LLM strategic intervention), making the system robust and scalable.
