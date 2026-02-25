# SpaceTraders Bot

An autonomous AI agent that plays the [SpaceTraders](https://spacetraders.io/) API game. The bot controls a fleet of ships, mines asteroids, fulfills contracts, and maintains a factual captain's log.

## Architecture

```
bot.py              # Main agent loop (LLM decision + tool execution)
tools.py            # 29 LangChain tools for game interaction
narrative.py        # Captain's log and strategic planning
ship_status.py      # Fleet cooldown/availability tracking
api_client.py       # SpaceTraders API wrapper
events.py           # Event logging
test_decisions.py   # Test harness for evaluating LLM decisions
tool_generator.py   # Utilities for API coverage analysis
```

### Key Design Decisions

1. **Custom agent loop** using LangChain's `bind_tools()` for structured tool calling, but with our own execution logic (not the default agent executor)

2. **Two distinct LLM calls per cycle:**
   - **DECIDE**: Given fleet state and tactical plan, choose next action(s)
   - **NARRATE**: After execution, record factual log entry

3. **Non-blocking ship operations**: Ships track their own cooldowns. If a ship is busy, the bot gets an error and can work with other ships or call `wait(seconds)` explicitly.

4. **Fleet state injection**: Current ship states (location, cargo, cooldowns) are injected before every decision.

5. **Tactical planning**: Strategic reflection generates actionable multi-step plans stored in `narrative_state.json`.

## Running

```bash
python bot.py              # Resume from saved session
python bot.py --fresh      # Start fresh, ignore saved state
python bot.py --clear      # Clear all saved state and exit
```

## Testing

Test the LLM's decision-making without running a live game:

```bash
python test_decisions.py                    # Run all test scenarios
python test_decisions.py --scenario=<name>  # Run specific scenario
python test_decisions.py --list             # List available scenarios
python test_decisions.py --model=<model>    # Test with different model
```

Test scenarios evaluate whether the LLM makes correct decisions in specific situations (e.g., "miner full, command ship nearby → should transfer cargo").

## State Files

- `session_state.json` - LLM message history for conversation continuity
- `narrative_state.json` - Mission state (goal, progress, tactical_plan, strategic_insight)
- `fleet_state.json` - Ship cooldowns and availability
- `story.jsonl` - Append-only captain's log
- `events.jsonl` - Append-only tool execution log

## Configuration

Environment variables (in `.env`):
- `TOKEN` - SpaceTraders API token (required)
- `MODEL` - Ollama model name (default: `glm-4.7-flash`)
- `OLLAMA_BASE_URL` - Ollama server URL (default: `http://192.168.1.171:11434`)

## Multi-Ship Coordination

The system prompt includes explicit tactical rules for fleet coordination:

1. **Depot pattern**: Park command ship at asteroid, miner transfers cargo to it
2. **Cooldown handling**: When one ship is busy, command another
3. **Cargo transfer**: `transfer_cargo` tool allows offloading from miner to command ship

The `transfer_cargo` tool requires both ships to be at the same waypoint and in ORBIT.

## Next Steps

### High Priority

1. **Validate test scenarios** - Run `python test_decisions.py` with different models to see which produce correct multi-ship decisions

2. **Handle API errors gracefully** - The SpaceTraders API can return cooldown errors directly; these should update the fleet tracker

3. **Try larger models** - The tactical decision rules may work better with more capable models

### Medium Priority

4. **Better fleet state sync** - Currently only updates when `view_ships` is called. Could refresh periodically or after each action.

5. **Market price tracking** - Store price history for arbitrage opportunities

6. **Ship purchasing triggers** - Add heuristics for when to consider buying more ships

### Low Priority / Ideas

7. **Multiple asteroid support** - Find and work multiple mining locations

8. **Multi-system expansion** - Discover and operate in multiple star systems

9. **Web UI** - Display the captain's log in a nicer format than terminal

## Code Notes

- Tools use `@tool` decorator from LangChain for automatic schema generation
- `WAITING_TOOLS` set identifies tools that have cooldowns (`navigate_ship`, `extract_ore`)
- `SIGNIFICANT_TOOLS` set identifies tools worth narrating (now includes `transfer_cargo`)
- Fleet tracker checks happen in bot loop before tool execution, not in tools themselves
- Fleet state is injected before every decision (not just every 5 iterations)
- Strategic reflection outputs `tactical_plan` (list of steps) in addition to insight
- Captain's log is now factual/dry rather than narrative prose
