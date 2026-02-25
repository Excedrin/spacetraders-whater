# SpaceTraders Bot Redesign: Integrated Narrative

## Overview

Redesign the bot to integrate narrative generation directly into its decision loop, giving it a persistent "high-level view" of the world, its goals, and what it's working toward.

## Key Design Decisions

1. **Custom agent loop** - Replace LangChain's `create_agent` with explicit control
2. **Two distinct LLM calls per cycle**:
   - **Decision call**: Given narrative context, choose next action(s)
   - **Narrative call**: After execution, generate narrative + reflection
3. **Same model** for both calls (coherent voice)
4. **Parallel narrative generation** during tool downtime (navigation, extraction cooldowns)
5. **Bot handles all output** - `narrator.py` becomes obsolete

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LOOP                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  NARRATIVE CONTEXT (~1000 tokens, in-memory)             │  │
│  │  - Current mission/goal                                  │  │
│  │  - Progress (quantified)                                 │  │
│  │  - Last 5-7 narrative segments                           │  │
│  │  - Current reflection/strategy                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. DECIDE (LLM Call #1)                                 │  │
│  │     Prompt: System + Narrative Context + Task            │  │
│  │     Output: Tool call(s) to execute                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. EXECUTE                                              │  │
│  │     Run tool(s), collect results                         │  │
│  │     If tool has downtime → run narrative in parallel     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. NARRATE (LLM Call #2) — if significant action        │  │
│  │     Prompt: Event data + context → narrative + reflect   │  │
│  │     Output: Narrative segment, updated goal/progress     │  │
│  │     (May already be complete if ran in parallel)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4. UPDATE CONTEXT                                       │  │
│  │     Add segment to rolling buffer, trim old              │  │
│  │     Persist to story.jsonl                               │  │
│  │     Display narrative in terminal                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                         (Loop to 1)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `bot.py` | Rewrite | Custom agent loop with two-call structure |
| `narrative.py` | New | NarrativeContext class, segment generation |
| `tools.py` | Refactor | Separate API calls from waits, return wait times |
| `events.py` | Keep | Raw event logging (useful for debugging) |
| `narrator.py` | Delete | Obsolete, functionality moves into bot |

---

## Implementation Details

### 1. Refactor tools.py

**Goal**: Tools that have waits should return `(result, wait_seconds)` instead of blocking.

**Changes**:
- Add `WaitingResult` dataclass: `(message: str, wait_seconds: float)`
- `navigate_ship` returns wait time from arrival calculation
- `extract_ore` returns cooldown time
- Other tools return `wait_seconds=0`
- Remove `_wait_with_log` from tool internals (move to agent loop)
- Remove `recall_memory` tool (replaced by automatic context injection)
- Convert from `@tool` decorators to plain functions (no longer using LangChain tools)

**Tool categories**:
- **Observation tools**: `view_agent`, `view_contracts`, `view_ships`, `view_cargo`, `find_waypoints`, `view_shipyard`, `view_market`
- **Action tools (no wait)**: `accept_contract`, `purchase_ship`, `orbit_ship`, `dock_ship`, `refuel_ship`, `sell_cargo`, `jettison_cargo`, `deliver_contract`, `fulfill_contract`
- **Action tools (with wait)**: `navigate_ship`, `extract_ore`

### 2. Create narrative.py

**NarrativeContext class**:
```python
@dataclass
class NarrativeSegment:
    timestamp: datetime
    tool_name: str
    narrative: str  # 2-3 sentences, dramatic

@dataclass
class NarrativeContext:
    current_goal: str           # e.g., "Deliver 73 ALUMINUM_ORE to X1-KD26-A1"
    progress: str               # e.g., "6/73 delivered"
    reflection: str             # e.g., "Mining is slow, may need to find richer asteroid"
    segments: list[NarrativeSegment]  # Last 5-7 segments

    def to_prompt_block(self) -> str:
        """Render ~1000 token context for injection into decision prompt"""

    def add_segment(self, segment: NarrativeSegment):
        """Add segment, trim to max 7"""

    def update_from_narrative_response(self, response: dict):
        """Update goal, progress, reflection from narrative LLM response"""

    def persist(self):
        """Write to story.jsonl"""

    @classmethod
    def load(cls) -> "NarrativeContext":
        """Load from story.jsonl on startup"""
```

**Narrative generation prompt**:
```
You are WHATER, a sentient mining machine with extraordinary taste in music.
You just took an action in the game. Narrate what happened and reflect on your strategy.

=== What Just Happened ===
{tool_results}

=== Current Context ===
Goal: {current_goal}
Progress: {progress}

Respond in JSON:
{
  "narrative": "2-3 vivid sentences in 1950s pulp sci-fi style",
  "current_goal": "Updated goal statement (or same if unchanged)",
  "progress": "Updated progress (quantified)",
  "reflection": "1-2 sentences on strategy/what you're thinking"
}
```

### 3. Rewrite bot.py

**Decision prompt structure**:
```
=== SYSTEM ===
You are an autonomous space trading agent playing SpaceTraders...
[Game rules, mechanics, tool descriptions]

=== WHATER'S CHRONICLE ===
{narrative_context.to_prompt_block()}

=== TASK ===
Decide what to do next. Call one or more tools.
```

**Agent loop pseudocode**:
```python
def run():
    context = NarrativeContext.load()

    while True:
        # 1. DECIDE
        decision_prompt = build_decision_prompt(context)
        response = llm.chat(decision_prompt, tools=TOOL_SCHEMAS)

        if response.is_done:
            break

        tool_calls = parse_tool_calls(response)

        # 2. EXECUTE (with parallel narrative for waiting tools)
        results = []
        narrative_future = None

        for call in tool_calls:
            result, wait_time = execute_tool(call)
            results.append(result)

            if wait_time > 5 and is_significant(call):
                # Start narrative generation in parallel
                narrative_future = executor.submit(
                    generate_narrative, results, context
                )
                # Wait for game cooldown
                wait_with_display(wait_time)

        # 3. NARRATE (may already be done from parallel execution)
        if any(is_significant(c) for c in tool_calls):
            if narrative_future:
                segment = narrative_future.result()
            else:
                segment = generate_narrative(results, context)

            # 4. UPDATE CONTEXT
            context.add_segment(segment)
            context.persist()
            display_narrative(segment)
```

### 4. Parallel execution for downtime

**Waiting tools**:
- `navigate_ship`: Wait = arrival time - now
- `extract_ore`: Wait = cooldown seconds

**Execution pattern**:
```python
import concurrent.futures

def execute_with_parallel_narrative(tool_call, context, executor):
    result, wait_seconds = execute_tool(tool_call)

    if wait_seconds > 5 and is_significant_action(tool_call):
        # Fire off narrative generation
        future = executor.submit(generate_narrative, result, context)

        # Wait for game (with progress display)
        wait_with_display(wait_seconds, tool_call.name)

        # Narrative should be ready
        return result, future.result()

    return result, None
```

### 5. Terminal display

**During execution**:
```
[DECIDE] Checking contracts and ships...
[TOOL] view_contracts → 1 active contract
[TOOL] view_ships → 3 ships, WHATER-3 at X1-KD26-A1

[DECIDE] Navigate to asteroid for mining
[TOOL] navigate_ship(WHATER-3, X1-KD26-CB5E)
       ⏳ Arriving in 87s...
       [Generating narrative...]
       ⏳ 72s remaining...
       ⏳ 57s remaining...
       ✓ Arrived at X1-KD26-CB5E

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ navigate_ship
The void yawned before WHATER-3 as its ion drives flared brilliant
blue against the cosmic dark! Ninety seconds of silent gliding
through the star-scattered expanse of X1-KD26, and the excavator
drone kissed the gravity well of asteroid CB5E.

📋 Goal: Deliver 73 ALUMINUM_ORE to X1-KD26-A1
📊 Progress: 6/73 delivered
💭 The asteroid awaits. Time to fill these cargo holds.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Narrative Context Example (~1000 tokens)

```
=== WHATER'S CHRONICLE ===

[Mission]
Deliver 73 ALUMINUM_ORE to X1-KD26-A1 for contract cm85abc123
Payment: 13,180 credits on completion

[Progress]
6/73 delivered (8%)
Credits: 138,403

[Recent Events]
• [3 min ago] The excavator's lasers bit deep into CB5E's ancient
  crust, yielding 4 precious units of ALUMINUM_ORE! The cargo hold
  hummed with anticipation—11 units now secured, 62 more to harvest.

• [2 min ago] WHATER-3 pivoted gracefully, abandoning the depleted
  vein. Course locked: the fuel depot at A2. Even machines must drink.

• [1 min ago] Docking clamps engaged at X1-KD26-A2 with a satisfying
  clunk. The fuel gauge climbed from critical 47 to a healthy 100.
  The asteroid field beckoned once more.

• [Now] Ion drives ignited. WHATER-3 surged back toward CB5E, cargo
  holds hungry for aluminum. The contract deadline loomed, but the
  machine felt no fear—only purpose.

[Reflection]
Mining yields have been mixed—lots of iron and quartz alongside the
aluminum. Strategy: jettison non-contract ores when cargo fills,
maximize aluminum extraction per trip. May need to check if there's
a richer asteroid in the system.
```

---

## Migration Steps

1. **Phase 1**: Refactor `tools.py`
   - Add `WaitingResult` type
   - Modify `navigate_ship` and `extract_ore` to return wait times
   - Convert tools to plain functions
   - Remove `recall_memory`

2. **Phase 2**: Create `narrative.py`
   - Implement `NarrativeContext` and `NarrativeSegment`
   - Implement narrative generation prompt
   - Implement persistence (story.jsonl)
   - Implement `to_prompt_block()`

3. **Phase 3**: Rewrite `bot.py`
   - Implement custom agent loop
   - Two-call structure (decide, narrate)
   - Tool execution with result collection
   - Context injection into decision prompt

4. **Phase 4**: Add parallel narrative
   - ThreadPoolExecutor for concurrent LLM calls
   - Wait-with-display function
   - Parallel narrative during navigation/extraction

5. **Phase 5**: Polish terminal output
   - Rich formatting for narrative display
   - Progress indicators during waits
   - Clean separation between tool output and narrative

6. **Phase 6**: Cleanup
   - Delete `narrator.py`
   - Update any imports
   - Test full flow

---

## Open Questions / Future Enhancements

1. **Tool calling format**: How does the LLM specify which tool to call? Options:
   - JSON function calling (if model supports it)
   - Structured text parsing
   - ReAct-style (Thought/Action/Observation)

2. **Multiple actions per turn**: Should bot be able to call multiple tools before narrating?

3. **Error recovery**: If narrative generation fails, continue anyway?

4. **Startup behavior**: On restart, should bot summarize story.jsonl to rebuild context, or just load last N segments?

5. **Context compression**: As game progresses, could periodically "summarize" old narrative into a denser form?
