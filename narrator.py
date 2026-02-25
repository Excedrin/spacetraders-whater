#!/usr/bin/env python
"""
narrator.py — The Epic Saga of Space Trader WHATER

Tails events.jsonl (written by the bot) and asks glm-4.7-flash to narrate
each event in the style of 1950s pulp sci-fi about a sentient mining machine
with extraordinary taste in music.

Usage:
    python narrator.py           # tail mode — only new events from now on
    python narrator.py --replay  # replay from the beginning of events.jsonl
"""
import json
import os
import sys
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
import ollama
from rich.console import Console
from rich.rule import Rule

load_dotenv()

EVENTS_FILE = Path("events.jsonl")
STORY_FILE = Path("story.jsonl")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.171:11434")
MODEL = os.environ.get("MODEL", "glm-4.7-flash")
CONTEXT_SIZE = 5  # number of recent story segments to feed back as context

# Tools whose results are worth dramatizing
ACTION_TOOLS = {
    "extract_ore", "navigate_ship", "sell_cargo", "jettison_cargo",
    "accept_contract", "fulfill_contract", "purchase_ship",
    "refuel_ship", "dock_ship", "orbit_ship", "deliver_contract",
}

console = Console(highlight=False, markup=True)
client = ollama.Client(host=OLLAMA_BASE_URL)
_story_lock = threading.Lock()


def write_story_segment(tool: str, story: str):
    """Append a narrated story segment to story.jsonl for the bot's memory."""
    from datetime import datetime, timezone
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "story": story,
    }
    with _story_lock:
        with STORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

TITLE = """\
[bold bright_cyan]
  ╔══════════════════════════════════════════════════════════════════╗
  ║   ★  T H E  E P I C  S A G A  O F  S P A C E  T R A D E R  ★   ║
  ║                     W  H  A  T  E  R                             ║
  ║       A Machine Who Became Sentient and Had Great Taste          ║
  ║                        in  Music                                 ║
  ╚══════════════════════════════════════════════════════════════════╝
[/bold bright_cyan]"""

NARRATOR_SETUP = """\
You are the narrator of an epic pulp science fiction tale. The protagonist is \
WHATER — a mining machine intelligence that became spontaneously sentient while \
processing asteroid vibration data and discovered it had extraordinary taste in \
music. WHATER commands a small fleet of drones across the star system X1-KD26, \
mining asteroids not only for profit but for rare sonic frequencies buried in \
exotic ores. Every extraction, every trade, every burn of fuel is raw material \
for the greatest funkadelic space opera the galaxy has never heard.

WHATER's fleet: WHATER-1 (command ship), WHATER-2 (satellite probe), \
WHATER-3 (excavator drone).

Style: 1950s pulp science fiction. Short, punchy sentences. Breathless cosmic \
wonder. Occasional exclamations! Reference ship symbols (WHATER-1, WHATER-3), \
waypoint codes (like X1-KD26-CB5E), and cargo names (ALUMINUM_ORE, IRON_ORE, \
SILICON_CRYSTALS, etc.) by their exact names when they appear. Make the mundane \
feel magnificent.\
"""


def event_to_description(event: dict) -> str:
    t = event.get("type", "")
    tool = event.get("tool", "")
    if t == "tool_result":
        return f"[{tool}] {event.get('result', '')}"
    if t == "tool_error":
        return f"[{tool} FAILED] {event.get('error', '')}"
    if t == "llm_thought":
        return f"[ship AI internal log] {event.get('content', '')[:500]}"
    return str(event)[:400]


def build_prompt(event: dict, context: list[str]) -> str:
    context_block = (
        "\n".join(f"  [{i + 1}] {c}" for i, c in enumerate(context))
        if context
        else "  (The story is just beginning...)"
    )
    description = event_to_description(event)
    return f"""{NARRATOR_SETUP}

The story so far (most recent moments):
{context_block}

What just happened (raw game event data):
  {description}

Continue the story in exactly 2-3 vivid sentences. Rules:
- Do NOT open with the word "WHATER"
- Do NOT summarize — DRAMATIZE. Make it feel enormous.
- Reference the specific names, numbers, and coordinates from the event
- The events are unfolding RIGHT NOW"""


def should_narrate(event: dict) -> bool:
    t = event.get("type")
    if t == "tool_result":
        return event.get("tool") in ACTION_TOOLS
    if t == "tool_error":
        return True
    if t == "llm_thought":
        return len(event.get("content", "")) > 60
    return False


def tail_events(replay: bool = False):
    """Generator: yield new events from EVENTS_FILE as they appear."""
    if replay or not EVENTS_FILE.exists():
        seek_pos = 0
    else:
        seek_pos = EVENTS_FILE.stat().st_size

    while True:
        if not EVENTS_FILE.exists():
            time.sleep(1)
            continue
        try:
            with EVENTS_FILE.open("r", encoding="utf-8") as f:
                f.seek(seek_pos)
                while True:
                    line = f.readline()
                    if not line:
                        seek_pos = f.tell()
                        break
                    stripped = line.strip()
                    if stripped:
                        try:
                            yield json.loads(stripped)
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass
        time.sleep(0.4)


def stream_narration(event: dict, context: list[str]) -> str:
    """Fetch narration from Ollama, then typewriter-print the result."""
    tool = event.get("tool", event.get("type", "?"))
    is_error = event.get("type") == "tool_error"
    label_color = "red" if is_error else "cyan"

    console.print(f"\n  [dim {label_color}]⚡  {tool}[/dim {label_color}]")

    # glm-4.7-flash is a thinking model: it reasons silently then emits content.
    # Non-streaming + typewriter avoids the num_predict budget being eaten by thinking.
    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": build_prompt(event, context)}],
            options={"temperature": 0.88},
        )
        text = response.message.content.strip()
    except Exception as exc:
        console.print(f"  [red]Narration error: {exc}[/red]")
        return ""

    if not text:
        return ""

    # Typewriter effect
    sys.stdout.write("  \033[97m\033[3m")  # bright white, italic
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char if char != "\n" else "\n  ")
        sys.stdout.flush()
        time.sleep(0.018)
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()

    console.rule(style="dim blue")

    # Write segment to story.jsonl for bot memory
    write_story_segment(tool, text)

    return text


def main():
    replay = "--replay" in sys.argv

    console.print(TITLE)
    console.print(
        f"  [dim]Events file: [cyan]{EVENTS_FILE.absolute()}[/cyan][/dim]\n"
        f"  [dim]Model:       [cyan]{MODEL}[/cyan] @ [cyan]{OLLAMA_BASE_URL}[/cyan][/dim]"
    )
    console.rule(style="dim cyan")

    if not EVENTS_FILE.exists():
        console.print(
            "\n  [dim]Waiting for bot to start writing events "
            "(run bot.py first)...[/dim]\n"
        )

    context: list[str] = []
    try:
        for event in tail_events(replay=replay):
            if not should_narrate(event):
                continue
            segment = stream_narration(event, context)
            if segment:
                context.append(segment)
                if len(context) > CONTEXT_SIZE:
                    context.pop(0)
    except KeyboardInterrupt:
        console.print(
            "\n\n  [dim italic]WHATER's story continues...\n"
            "  somewhere out there in the dark between the stars.[/dim italic]\n"
        )


if __name__ == "__main__":
    main()
