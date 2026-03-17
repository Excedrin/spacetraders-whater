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

**HQ Fleet Director (v1.4+):**
- **Toggleable Autonomy:** `toggle_hq` tool enables/disables automated fleet management (in-memory flag, no persistence needed).
- **DRY Strategy Engine:** `evaluate_fleet_strategy()` — Single source of truth for game phase, budget, and fleet scaling decisions.
- **Four-Phase Progression:**
  1. **BOOTSTRAP** (<2 traders): Get first Hauler, basic autotrade
  2. **BUILDUP** (<3 traders): Buy second Hauler, saturate local market
  3. **GATE CONSTRUCTION** (3+ traders, gate incomplete): Slow-mode gate funding (1/3 of haulers) + autotrade
  4. **EXPANSION** (gate complete): Multi-system exploration, expand to 15 traders
- **Smart Satellite Routing:** `_get_probe_plan()` three-tier priority:
  1. Chart uncharted waypoints (triggers explore)
  2. Refresh oldest markets (local scouting)
  3. Jump to unexplored systems (Phase 4 only, when local markets <30min old)
- **Shipyard Squatting:** When fleet can expand, satellites automatically park at cheapest shipyard until purchase completes.
- **Improved Fleet Distribution:** `assign_idle_ships()` with reality checks:
  - Only gate construction when gate exists AND needs materials AND excess > 150k
  - Consistent hash assignment (same ships always do construction duty)
  - Haulers default to autotrade when no construction work
- **Adaptive Exploration:** `_step_explore()` now auto-jumps between systems:
  - When system fully charted/scouted, checks jump gate for unexplored connections
  - Automatically chains `goto GATE, jump WAYPOINT, explore` for continuous exploration
  - Stops exploring only when all connected space is fully mapped

**Advisor Status in Server State:**
- `GET /api/state` now includes `"advisor"` field with current financial assessment.
- Eliminates need for separate tool calls; atomic consistency across bot/CLI.
- `play_cli.py` reads advisor from state instead of calling tool directly.

**Trading Margin Fixes (v1.5+):**
- **Global Constants:** `TRADE_PROFIT_MARGIN = 0.10` and `JIT_TRADE_MARGIN = 0.02` replace hardcoded values throughout.
- **Cargo Cost Tracking:** When selling cargo you already own, `_build_sell_sequence()` now uses actual `cargo_cost` paid instead of market prices.
- **HQ JIT Planner Fix:** Critical bug fix — now properly decrements `spare_capacity` as trades are added, preventing overbooking (previously would queue 3 trades of 50 units each when only 100 units were available).
- **Enhanced Logging:** `_evaluate_hq_opportunities()` now logs each decision point (debug for rejections, info for insertions) and limits to one step insertion per cycle.

**Construct Step Improvements (v1.5+):**
- **Auto-Sell Unwanted Cargo:** `_step_construct()` no longer pauses when encountering wrong cargo; instead calls `_build_sell_sequence()` to automatically sell it, then re-queue construct.
- **Fleet-Aware Prevention:** HQ idle assignment now checks `_find_active_ships_with_keywords(['supply', 'construct'])` before assigning another ship to construction, preventing duplicate material gathering.
- **Idle on Overflow:** If another ship is already delivering enough material, the construct step completes gracefully and lets the ship go idle (available for HQ reassignment).

**Fleet-Aware Helper Functions (v1.5+):**
- `_find_active_ships_with_keywords(keywords, exclude_ship)` — Returns set of ships actively running behaviors with given keywords (e.g., 'construct', 'supply').
- `_extract_active_goods_for_keyword(keyword)` — Parses fleet behaviors to extract goods being traded with specific keywords (e.g., extract 'IRON_ORE' from 'buy IRON_ORE').
- Used by both `autotrade` (to avoid duplicate trading) and `construct` (to avoid duplicate assignments).

**Shared Sell Sequence Builder:**
- `_build_sell_sequence(ship_symbol, cargo_map, ship)` — Extracted from trade route planner into reusable function. Finds best sell market for each cargo item, groups by destination, returns navigation/sell steps. Handles gate material special case (supply to jump gate if materials needed).
- Used by both `_plan_trade_route()` (initial cargo clearance) and `_step_construct()` (clearing unwanted cargo before proceeding).

**Utility Helpers (v1.6+):**
- `get_system_from_waypoint(waypoint_symbol)` — Centralized string manipulation to extract system symbol (e.g., `'X1-DF14'` from `'X1-DF14-A1'`). Used throughout codebase for reliable waypoint parsing.
- `calculate_distance(x1, y1, x2, y2)` and `waypoint_distance(wp1, wp2, cache)` — Centralized Euclidean distance functions used by pathfinding, probe planning, and trade analysis. Eliminates hardcoded math across multiple files.
- Reduces code duplication: these helpers eliminated a dozen instances of manual string splitting and distance calculation.

**HQ Design Philosophy — Short-Lived Behaviors + Idle Re-evaluation:**
The core pattern is that ships receive short, focused behaviors (e.g., `goto WP, scout, stop` or a single trade route ending in `stop`). When a behavior completes, the ship becomes idle, and `assign_idle_ships()` re-evaluates what it should do next based on current game state. This means strategic pivots happen naturally — a hauler that just finished a trade run might be reassigned to buy construction materials, or a probe that just scouted a market might be sent to the most stale market across the entire system. No behavior needs to encode long-term strategy; the HQ director handles that by making fresh decisions each time a ship goes idle. This keeps individual behaviors simple while enabling complex fleet-wide coordination like "pause trading to fund jump gate construction" or "redirect probes to cover neglected market clusters."

This design successfully separates the "Game Engine" (autonomous server tick loop) from the "Brain" (occasional LLM strategic intervention), making the system robust and scalable. The HQ Director extends this by adding a third layer: **Autonomous Task Distribution** that intelligently assigns idle ships based on game phase, without requiring LLM intervention.
