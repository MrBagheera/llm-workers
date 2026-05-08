"""Slack session state management backed by local JSON files."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict

SESSIONS_DIR = Path(os.environ.get("SLACK_SESSIONS_DIR", "."))


def make_session_id(channel_id: str, thread_ts: str) -> str:
    """Build a stable session ID from channel and thread timestamp."""
    return f"{channel_id}__{thread_ts.replace('.', '_')}"


@dataclass
class SlackSession:
    session_id: str
    thread_ts: str
    channel_id: str
    is_processing: bool = False
    awaiting_confirmation: bool = False
    messages: list[BaseMessage] = field(default_factory=list)
    pending_tool_call_ids: list[str] = field(default_factory=list)
    thinking_message_ts: str = ""

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, session_id: str) -> Optional["SlackSession"]:
        """Load session from disk. Returns None if the file does not exist."""
        path = _session_path(session_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        messages = messages_from_dict(data.pop("messages", []))
        return cls(messages=messages, **data)

    @classmethod
    def create(cls, channel_id: str, thread_ts: str) -> "SlackSession":
        """Create a new empty session and persist it immediately."""
        session_id = make_session_id(channel_id, thread_ts)
        session = cls(session_id=session_id, thread_ts=thread_ts, channel_id=channel_id)
        session.save()
        return session

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically write session state to disk."""
        path = _session_path(self.session_id)
        tmp = path.with_suffix(".tmp")
        data = {
            "session_id": self.session_id,
            "thread_ts": self.thread_ts,
            "channel_id": self.channel_id,
            "is_processing": self.is_processing,
            "awaiting_confirmation": self.awaiting_confirmation,
            "messages": [message_to_dict(m) for m in self.messages],
            "pending_tool_call_ids": self.pending_tool_call_ids,
            "thinking_message_ts": self.thinking_message_ts,
        }
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"
