This is a significant architectural shift from a **Centralized Control** model (God-mode LLM micromanaging every click) to a **Hierarchical Control** model (LLM as Fleet Admiral, code as Ship Captains).

Here is the design plan to implement automated state machine loops.

### 1. The Core Concept: "Behaviors"
Instead of the LLM calling `navigate`, `extract`, and `sell` individually, we introduce the concept of **Behaviors** (or "Jobs").

A **Behavior** is a Python class or a structured state machine that:
1.  Owns a specific ship.
2.  Maintains internal state (e.g., `MINING`, `TRANSIT_TO_SELL`, `SELLING`).
3.  Has a `tick()` method called every game loop.
4.  Persists its configuration to disk (so loops survive restarts).

The LLM’s role changes from **Operator** to **Orchestrator**. It simply assigns a Behavior and monitors for failure.

---

### 2. Data Structure Changes

#### A. `ship_behavior.py` (New Module)
You need a system to define these loops. Since you want the LLM to "produce" them, you shouldn't hardcode *too* much logic, but you need safety. I recommend a **Parametrized State Machine** approach.

**The State Machine Structure:**
The LLM will not write Python code. It will call a tool that saves a JSON configuration to the ship.

**Example JSON Configuration (Mental Model):**
```json
{
  "ship_symbol": "WHATER-1",
  "behavior_type": "LOOP_RESOURCE_CHAIN",
  "parameters": {
    "primary_location": "X1-AB12-A1",   # Asteroid
    "secondary_location": "X1-AB12-B2", # Market
    "primary_action": "EXTRACT",
    "secondary_action": "SELL",
    "target_commodities": ["IRON_ORE", "COPPER_ORE"],
    "min_fuel_threshold": 100
  },
  "current_state": "AT_PRIMARY"
}
```

#### B. `FleetTracker` Update
Your `FleetTracker` in `ship_status.py` currently tracks passive status (fuel, cooldown). It needs to be upgraded to track **Intent**.
*   Add a `behavior: Optional[dict]` field to `ShipStatus`.
*   This ensures that when the bot restarts, it knows "WHATER-1 is supposed to be mining iron," not just "WHATER-1 is at an asteroid."

---

### 3. The "Tactical Loop" (The Engine)

You need to split your `bot.py` execution into two layers:

1.  **The Strategic Layer (LLM):** Runs every ~10-30 seconds or when an event occurs. Checks fleet status, analyzes markets, updates plans.
2.  **The Tactical Layer (Code):** Runs every 1-2 seconds (fast). Iterates through all ships and executes one step of their assigned Behavior.

**How the Tactical Layer works:**
It looks at the `behavior_type` and executes hard-coded logic for that state.

**State Machine Logic (Pseudocode Design):**

**Type: `MINING_LOOP`**
*   **State: TRAVEL_TO_MINE**
    *   If at location: Switch to `EXTRACT`.
    *   Else: Check fuel. If low -> `REFUEL`. Else -> `navigate(asteroid)`.
*   **State: EXTRACT**
    *   If cooldown > 0: Wait.
    *   If cargo full: Switch to `TRAVEL_TO_SELL`.
    *   Else: `extract()`. Then `jettison(unwanted)`.
*   **State: TRAVEL_TO_SELL**
    *   If at location: Switch to `SELL`.
    *   Else: `navigate(market)`.
*   **State: SELL**
    *   `sell_cargo(targets)`.
    *   If cargo empty (of targets): Switch to `TRAVEL_TO_MINE`.

---

### 4. New Tools for the LLM

The LLM needs high-level tools to configure these machines. Remove the pressure to use `navigate` manually.

1.  **`assign_mining_loop(ship_symbol, asteroid_wp, market_wp, ore_types)`**
    *   Validates that the ship can mine.
    *   Validates that the market actually buys the ore (using your cache).
    *   Sets the ship's state in `FleetTracker`.

2.  **`assign_trade_route(ship_symbol, buy_wp, sell_wp, trade_good)`**
    *   Sets the ship to buy low at A and sell high at B repeatedly.

3.  **`assign_static_task(ship_symbol, task_type)`**
    *   e.g., `SATELLITE_SCOUT` (randomly orbits within system to keep market data fresh).
    *   e.g., `STATIC_DRILL` (Mine -> Jettison non-targets -> Repeat).

4.  **`cancel_assignment(ship_symbol)`**
    *   Clears the behavior. Returns the ship to manual control (IDLE).

---

### 5. Handling Interruptions & Errors (The Feedback Loop)

This is crucial. The automated loop *will* fail (e.g., market crashes, pirate attack, out of fuel).

**The Event System:**
When the Tactical Layer encounters an error it cannot solve (e.g., tried to refuel but no credits, or market refuses trade):
1.  It sets the ship's Behavior to `SUSPENDED` or `ERROR`.
2.  It pushes a high-priority event to the `events.jsonl` (or a new alert queue).
3.  The next time the LLM runs, it sees:
    *   `[ALERT] WHATER-1 Mining Loop Failed: Insufficient credits to refuel at X1-AB12-B2.`

**LLM Response:**
The LLM sees the alert, uses `view_agent` to check credits, maybe `sell_cargo` on another ship to raise funds, then calls `resume_assignment(WHATER-1)`.

---

### 6. Implementation Stages

#### Phase 1: The Idle Miner
Implement the "Static Mining Loop" first.
*   **Goal:** Ship sits at asteroid. Mines. Jettisons waste. Keeps specific ore.
*   **Reasoning:** No navigation logic required. Easiest to test state transitions (Cooldown -> Action -> Jettison).

#### Phase 2: The Tactical Engine
Modify `bot.py` to decouple the LLM from the loop.
*   The script should run a `while True` loop.
*   Inside, checking ship behaviors happens frequently.
*   Calling the LLM happens only when a ship is `IDLE` or needs strategic input.

#### Phase 3: Navigation Integration
Add the `navigate` logic to the state machine.
*   **Complexity:** Must handle `fuel` checks automatically using the logic already present in your `tools.py`.
*   **Safety:** The loop must automatically insert a `REFUEL` state if fuel < required for the trip.

#### Phase 4: LLM Tooling
Expose the tools to the LLM. Update the System Prompt to explain:
*   "Do not micromanage ships."
*   "Assign loops using `assign_mining_loop`."
*   "Monitor their status in the [Fleet Status] block."

### Summary of Benefits

1.  **Token Efficiency:** You stop streaming "Navigate... Wait... Extract... Wait..." into the context window.
2.  **Speed:** The bot reacts to cooldowns instantly (ms), not waiting for an LLM inference (seconds).
3.  **Stability:** Logic defined in code (fuel checks, cargo checks) is less hallucination-prone than logic defined in LLM generation.
4.  **Scalability:** You can manage 50 ships as easily as 2. The LLM manages the *Fleet Strategy*, the code manages the *Ship Logistics*.
