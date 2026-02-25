import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

EVENTS_FILE = Path("events.jsonl")
_lock = threading.Lock()


def write_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with _lock:
        with EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")


class EventLogger(BaseCallbackHandler):
    """LangChain callback that writes game events to events.jsonl for the narrator."""

    def __init__(self):
        self._pending: dict[str, str] = {}  # run_id -> tool_name

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "unknown")
        self._pending[str(run_id)] = name

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = self._pending.pop(str(run_id), "unknown")
        result = output.content if hasattr(output, "content") else str(output)
        write_event({"type": "tool_result", "tool": name, "result": result})

    def on_tool_error(self, error, *, run_id, **kwargs):
        name = self._pending.pop(str(run_id), "unknown")
        write_event({"type": "tool_error", "tool": name, "error": str(error)})

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg:
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and len(content.strip()) > 40:
                        write_event({"type": "llm_thought", "content": content.strip()})
