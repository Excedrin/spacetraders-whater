"""
narrative.py — Narrative context management for the SpaceTraders bot.

Maintains a rolling window of narrative segments that get injected into
the bot's decision prompt, giving it a persistent "high-level view" of
the world, its goals, and what it's working toward.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import ollama

load_dotenv()

STORY_FILE = Path("story.jsonl")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.171:11434")
MODEL = os.environ.get("MODEL", "glm-4.7-flash")

MAX_SEGMENTS = 7  # Rolling window size
CONTEXT_TOKEN_BUDGET = 1000  # Approximate target

ollama_client = ollama.Client(host=OLLAMA_BASE_URL)


NARRATOR_PERSONA = """\
You are WHATER, a fleet coordinator AI writing a factual captain's log.

Write DRY, FACTUAL entries. Record:
- What action was taken
- Current resource state (cargo, fuel, credits)
- What should happen next

Be CLINICAL and PRECISE. No flowery prose. Example:
"Extracted 3 ICE_WATER. WHATER-3 cargo now 12/15. Should transfer to WHATER-1 before next extraction."
"""


@dataclass
class NarrativeSegment:
    """A single narrative moment in WHATER's story."""
    timestamp: datetime
    tool_name: str
    narrative: str  # 2-3 dramatic sentences

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp.isoformat(),
            "tool": self.tool_name,
            "story": self.narrative,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NarrativeSegment":
        return cls(
            timestamp=datetime.fromisoformat(data["ts"]),
            tool_name=data.get("tool", "?"),
            narrative=data.get("story", ""),
        )


@dataclass
class NarrativeContext:
    """
    The bot's narrative memory - captain's log only.

    Strategic planning is handled by plan.txt (updated via strategic reflection).
    This class only maintains the factual captain's log segments.
    """
    segments: list[NarrativeSegment] = field(default_factory=list)
    chapter: int = 1  # Current chapter number
    chapter_title: str = "Awakening"  # Current chapter title

    def to_prompt_block(self) -> str:
        """Render recent captain's log entries (if any)."""
        if not self.segments:
            return ""

        lines = ["[Recent Log]"]
        for seg in self.segments[-3:]:  # Last 3 entries
            lines.append(f"• {seg.narrative}")

        return "\n".join(lines)

    def _format_age(self, ts: datetime) -> str:
        """Format timestamp as relative age."""
        now = datetime.now(timezone.utc)
        delta = now - ts
        minutes = int(delta.total_seconds() / 60)
        if minutes < 1:
            return "Now"
        elif minutes == 1:
            return "1 min ago"
        elif minutes < 60:
            return f"{minutes} min ago"
        else:
            hours = minutes // 60
            return f"{hours}h ago"

    def add_segment(self, segment: NarrativeSegment):
        """Add a new segment, trimming old ones to stay within budget."""
        self.segments.append(segment)
        if len(self.segments) > MAX_SEGMENTS:
            self.segments = self.segments[-MAX_SEGMENTS:]

    def update_from_response(self, response: dict):
        """Update chapter info from narrative generation response."""
        if "new_chapter" in response and response["new_chapter"]:
            self.chapter += 1
            if "chapter_title" in response:
                self.chapter_title = response["chapter_title"]

    def persist(self):
        """Append latest segment to story.jsonl."""
        if not self.segments:
            return
        latest = self.segments[-1]
        with STORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(latest.to_dict()) + "\n")

    def persist_full(self):
        """Write full context state (for recovery)."""
        state = {
            "chapter": self.chapter,
            "chapter_title": self.chapter_title,
            "segments": [s.to_dict() for s in self.segments],
        }
        state_file = Path("narrative_state.json")
        state_file.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls) -> "NarrativeContext":
        """Load context from story.jsonl on startup."""
        context = cls()

        # Try to load full state first
        state_file = Path("narrative_state.json")
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                context.chapter = state.get("chapter", 1)
                context.chapter_title = state.get("chapter_title", "Awakening")
                context.segments = [
                    NarrativeSegment.from_dict(s)
                    for s in state.get("segments", [])
                ]
                return context
            except (json.JSONDecodeError, KeyError):
                pass

        # Fall back to loading from story.jsonl
        if STORY_FILE.exists():
            try:
                lines = STORY_FILE.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-MAX_SEGMENTS:]:
                    try:
                        data = json.loads(line)
                        context.segments.append(NarrativeSegment.from_dict(data))
                    except json.JSONDecodeError:
                        pass
            except OSError:
                pass

        return context


def generate_narrative(
    tool_results: list[tuple[str, str]],
    context: NarrativeContext,
    model: str = MODEL,
    fleet_state: str = "",
) -> Optional[NarrativeSegment]:
    """
    Generate a narrative segment for recent tool actions.

    Args:
        tool_results: List of (tool_name, result_message) tuples
        context: Current narrative context
        model: Model to use for generation
        fleet_state: Current fleet status (ships, locations, cargo) - optional

    Returns:
        NarrativeSegment or None if generation fails
    """
    # Format what just happened
    events_block = "\n".join(
        f"[{name}] {result[:300]}" for name, result in tool_results
    )

    # Build story context from recent segments
    if context.segments:
        story_so_far = "\n".join(
            f"• {seg.narrative}" for seg in context.segments[-3:]
        )
    else:
        story_so_far = "(This is the beginning of your story.)"

    # Include fleet state if provided
    fleet_section = f"\nYOUR FLEET:\n{fleet_state}\n" if fleet_state else ""

    prompt = f"""{NARRATOR_PERSONA}

=== Log Entry ===
{fleet_section}
PREVIOUS LOG ENTRIES:
{story_so_far}

ACTIONS JUST COMPLETED:
{events_block}

Record this in the captain's log. Be factual and brief.

Respond in JSON format:
{{
  "narrative": "1-2 factual sentences. State what happened and current state."
}}"""

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8},
        )
        text = response.message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)

        # Create segment
        segment = NarrativeSegment(
            timestamp=datetime.now(timezone.utc),
            tool_name=tool_results[-1][0] if tool_results else "unknown",
            narrative=data.get("narrative", ""),
        )

        # Update context with goal/progress/reflection
        context.update_from_response(data)

        return segment

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [narrative] Failed to parse response: {e}")
        return None
    except Exception as e:
        print(f"  [narrative] Generation error: {e}")
        return None


def generate_strategic_reflection(
    context: NarrativeContext,
    game_state: str,
    model: str = MODEL,
) -> tuple[Optional[NarrativeSegment], Optional[dict]]:
    """
    Generate a deep strategic reflection every ~10 cycles.

    This is where WHATER steps back and thinks about the bigger picture:
    - Are there better approaches?
    - How can the fleet be used more efficiently?
    - What patterns have emerged?
    - What's the story arc?

    Args:
        context: Current narrative context
        game_state: Current game state (ships, cargo, contracts, etc.)
        model: Model to use for generation

    Returns:
        Tuple of (NarrativeSegment, response_dict) or (None, None) if generation fails
        The response_dict contains the full JSON including recommended_plan
    """
    # Build a summary of recent narrative
    recent_narrative = "\n".join(
        f"• {seg.narrative}" for seg in context.segments[-5:]
    ) if context.segments else "(No recent events)"

    prompt = f"""{NARRATOR_PERSONA}

=== STRATEGIC REVIEW ===

CURRENT GAME STATE:
{game_state}

RECENT ACTIVITY:
{recent_narrative}

Analyze the situation and create an actionable plan.

ANALYSIS QUESTIONS:
1. Fleet utilization: Are all ships being used? Any sitting idle?
2. Bottlenecks: What's slowing progress? (cargo full, waiting on cooldowns, travel time)
3. Opportunities: Better routes? Market prices? Unexplored waypoints?
4. Efficiency: Could ships be positioned better? Should command ship be at asteroid?

OUTPUT FORMAT (JSON):
{{
  "narrative": "2-3 sentences. Factual assessment of current situation.",
  "recommended_plan": "Action plan with numbered steps (see rules below)",
  "new_chapter": false,
  "chapter_title": "Only if new_chapter is true"
}}

CRITICAL RULES for recommended_plan:
- Write ONLY actionable steps (what to DO next)
- DO NOT include current state (ship positions, fuel levels, cargo contents, distances)
- DO NOT repeat information already in game state sections
- Focus on GOALS and ACTIONS, not status documentation
- Keep it concise - the bot has full context in game state

Good example:
"1. Use DRIFT mode to reach nearest fuel station
2. Refuel both ships
3. Navigate to asteroid and mine aluminum ore
4. Deliver to contract waypoint"

Bad example (DO NOT DO THIS):
"WHATER-1 @ B13 with 97/400 fuel, needs 248 more to reach B6 which is 315.7 away, cargo: 1 ALUMINUM_ORE..."
This is BAD because it's documenting state instead of planning actions.

Set new_chapter=true only if: completed a major contract, acquired new ship, or discovered new system."""

    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.9},  # Slightly higher for creative thinking
        )
        text = response.message.content.strip()

        # Parse JSON response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)

        # Create segment marked as strategic reflection
        segment = NarrativeSegment(
            timestamp=datetime.now(timezone.utc),
            tool_name="strategic_reflection",
            narrative=data.get("narrative", ""),
        )

        # Update context with all fields including strategic insight
        context.update_from_response(data)

        return segment, data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [strategic] Failed to parse response: {e}")
        return None, None
    except Exception as e:
        print(f"  [strategic] Generation error: {e}")
        return None, None
