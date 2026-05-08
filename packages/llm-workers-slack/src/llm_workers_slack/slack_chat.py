"""LLM orchestration and Slack message-posting logic for a single processing run."""

import asyncio
import logging
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from slack_sdk.web.async_client import AsyncWebClient

from llm_workers.api import (
    ConfirmationRequest,
    ConfirmationResponse,
    WorkerNotification,
)
from llm_workers.worker import Worker
from llm_workers_slack.session import SlackSession

logger = logging.getLogger(__name__)

_LOG_TRUNCATE = 39_000


class SlackChat:
    """Handles one round of LLM interaction for a Slack thread."""

    def __init__(self, client: AsyncWebClient, worker: Worker, channel_id: str):
        self._client = client
        self._worker = worker
        self._channel_id = channel_id

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def run(self, session: SlackSession, user_message: str) -> None:
        """Standard chat flow: append user message, stream worker, post result."""
        session.messages.append(HumanMessage(content=user_message))

        await self._post_thinking(session)

        ai_messages, confirmation = await self._stream_worker(session)

        if confirmation is not None:
            await self._handle_confirmation_request(session, confirmation, ai_messages)
        else:
            await self._post_ai_response(session, ai_messages)
            session.messages.extend(ai_messages)
            session.is_processing = False
            session.save()

    async def resume_from_confirmation(self, session: SlackSession, approved: bool) -> None:
        """Resume processing after the user clicks Yes or No on a confirmation button."""
        approved_ids = session.pending_tool_call_ids if approved else []
        confirmation_response = ConfirmationResponse(approved_tool_calls=approved_ids)

        session.pending_tool_call_ids = []
        await self._post_thinking(session)

        messages_with_response = session.messages + [confirmation_response]
        ai_messages, confirmation = await self._stream_worker(session, extra=confirmation_response)

        if confirmation is not None:
            await self._handle_confirmation_request(session, confirmation, ai_messages)
        else:
            await self._post_ai_response(session, ai_messages)
            session.messages.extend(ai_messages)
            session.is_processing = False
            session.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_thinking(self, session: SlackSession) -> None:
        result = await self._client.chat_postMessage(
            channel=self._channel_id,
            thread_ts=session.thread_ts,
            text="Thinking...",
        )
        session.thinking_message_ts = result["ts"]
        session.save()

    async def _stream_worker(
        self,
        session: SlackSession,
        extra: Optional[ConfirmationResponse] = None,
    ) -> tuple[list[BaseMessage], Optional[ConfirmationRequest]]:
        """Run Worker.stream() in a thread pool. Returns (collected messages, confirmation or None)."""
        messages = session.messages if extra is None else session.messages + [extra]

        worker = self._worker
        collected, confirmation, tool_log = await asyncio.to_thread(
            _run_worker_sync, worker, messages
        )

        # Update the Thinking... message with the tool execution log
        log_text = tool_log[:_LOG_TRUNCATE]
        update_text = f"```\n{log_text}\n```" if log_text else "_No tool activity._"
        try:
            await self._client.chat_update(
                channel=self._channel_id,
                ts=session.thinking_message_ts,
                text=update_text,
            )
        except Exception:
            logger.warning("Failed to update Thinking... message", exc_info=True)

        return collected, confirmation

    async def _post_ai_response(
        self, session: SlackSession, ai_messages: list[BaseMessage]
    ) -> None:
        """Post the final AI text response to the Slack thread."""
        text = _extract_text(ai_messages)
        if not text:
            return
        await self._client.chat_postMessage(
            channel=self._channel_id,
            thread_ts=session.thread_ts,
            text=text,
        )

    async def _handle_confirmation_request(
        self,
        session: SlackSession,
        request: ConfirmationRequest,
        pre_msgs: list[BaseMessage],
    ) -> None:
        """Send a Block Kit Yes/No prompt and suspend processing until a button is clicked."""
        session.messages.extend(pre_msgs)
        session.pending_tool_call_ids = list(request.tool_calls.keys())

        confirmation_text = _format_confirmation_text(request)
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": confirmation_text[:3000]},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Yes"},
                        "style": "primary",
                        "action_id": "confirm_yes",
                        "value": session.session_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "No"},
                        "style": "danger",
                        "action_id": "confirm_no",
                        "value": session.session_id,
                    },
                ],
            },
        ]
        await self._client.chat_postMessage(
            channel=self._channel_id,
            thread_ts=session.thread_ts,
            text=confirmation_text[:3000],
            blocks=blocks,
        )

        session.is_processing = False
        session.awaiting_confirmation = True
        session.save()


# ------------------------------------------------------------------
# Module-level helpers (no self, safe to run in thread pool)
# ------------------------------------------------------------------

def _run_worker_sync(
    worker: Worker,
    messages: list,
) -> tuple[list[BaseMessage], Optional[ConfirmationRequest], str]:
    """
    Consume Worker.stream() synchronously (runs in a thread pool).
    Returns (collected BaseMessages, ConfirmationRequest or None, tool log string).
    """
    collected: list[BaseMessage] = []
    confirmation: Optional[ConfirmationRequest] = None
    log_lines: list[str] = []

    for item_list in worker.stream(messages):
        item = item_list[0]
        if isinstance(item, WorkerNotification):
            if item.type == "tool_start" and item.text:
                log_lines.append(f"+ {item.text}")
            elif item.type == "tool_end":
                log_lines.append(f"  done")
        elif isinstance(item, ConfirmationRequest):
            confirmation = item
            break
        elif isinstance(item, BaseMessage):
            collected.append(item)

    return collected, confirmation, "\n".join(log_lines)


def _extract_text(messages: list[BaseMessage]) -> str:
    """Extract displayable text from a list of AI messages."""
    parts = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            if content.strip():
                parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text.strip():
                        parts.append(text)
    return "\n\n".join(parts)


def _format_confirmation_text(request: ConfirmationRequest) -> str:
    """Build a human-readable description of what the AI wants to do."""
    lines = []
    for desc in request.tool_calls.values():
        lines.append(f"*AI wants to {desc.action}*")
        for param in desc.params:
            value = str(param.value)
            if param.format:
                lines.append(f"• `{param.name}`:\n```{param.format}\n{value}\n```")
            else:
                lines.append(f"• `{param.name}`: {value}")
    return "\n".join(lines)
