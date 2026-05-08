"""Entry point and Slack Bolt event handlers for llm-workers-slack."""

import argparse
import asyncio
import logging
import os
import re
import sys
from functools import partial
from typing import Optional

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from llm_workers.starlark import EvaluationContext
from llm_workers.user_context import StandardUserContext
from llm_workers.utils import setup_logging_from_args, add_common_logging_args
from llm_workers.worker import Worker
from llm_workers.worker_utils import ensure_env_vars_defined
from llm_workers.workers_context import StandardWorkersContext
from llm_workers_slack.session import SlackSession, make_session_id
from llm_workers_slack.slack_chat import SlackChat

logger = logging.getLogger(__name__)


def main():
    """Entry point for the llm-workers-slack command."""
    parser = argparse.ArgumentParser(
        description="Slack Socket Mode bot for LLM Workers scripts."
    )
    add_common_logging_args(parser)
    parser.add_argument(
        "script_file",
        type=str,
        help="Path to the LLM script file (supports module:resource.yaml syntax).",
    )
    args = parser.parse_args()

    log_file = setup_logging_from_args(args, log_filename="llm-workers-slack.log")
    print(f"Logging to {log_file}", file=sys.stderr)

    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not bot_token or not app_token:
        parser.error(
            "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in the environment "
            "or in ~/.config/llm-workers/.env"
        )

    user_config = StandardUserContext.load_config()
    environment = EvaluationContext.default_environment()
    ensure_env_vars_defined(environment, user_config.env)
    user_context = StandardUserContext(user_config, environment)

    script = StandardWorkersContext.load_script(args.script_file)
    workers_context = StandardWorkersContext(script, user_context)

    workers_context.run_async(_async_main, workers_context, bot_token, app_token)


async def _async_main(
    workers_context: StandardWorkersContext,
    bot_token: str,
    app_token: str,
) -> None:
    if not workers_context.config.chat:
        raise ValueError("'chat' section is missing from the script file")

    app = AsyncApp(token=bot_token)
    worker = Worker(workers_context.config.chat, workers_context, scope="chat")

    _register_handlers(app, worker)

    print("Starting Slack bot...", file=sys.stderr)
    handler = AsyncSocketModeHandler(app, app_token)
    await handler.start_async()


def _register_handlers(app: AsyncApp, worker: Worker) -> None:
    @app.event("app_mention")
    async def handle_mention(event, say, client):
        await _on_mention(event, say, client, worker)

    @app.action("confirm_yes")
    async def handle_confirm_yes(ack, body, client):
        await ack()
        await _on_button(body, client, worker, approved=True)

    @app.action("confirm_no")
    async def handle_confirm_no(ack, body, client):
        await ack()
        await _on_button(body, client, worker, approved=False)


async def _on_mention(event, say, client, worker: Worker) -> None:
    channel_id = event["channel"]
    thread_ts: Optional[str] = event.get("thread_ts")
    event_ts: str = event["ts"]
    text = re.sub(r"^<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()

    if thread_ts is None:
        # Top-level mention → start a new session
        session = SlackSession.create(channel_id, event_ts)
    else:
        # Threaded mention → look up existing session
        session = SlackSession.load(make_session_id(channel_id, thread_ts))
        if session is None:
            await say(
                text="Please start a new chat session by @-mentioning me in a top-level message.",
                thread_ts=thread_ts,
            )
            return

    if session.is_processing:
        await say(
            text="Already thinking on a previous message, ignoring this.",
            thread_ts=session.thread_ts,
        )
        return

    if session.awaiting_confirmation:
        await say(
            text="Please answer the pending confirmation above before continuing.",
            thread_ts=session.thread_ts,
        )
        return

    session.is_processing = True
    session.save()

    slack_chat = SlackChat(client, worker, channel_id)
    asyncio.create_task(
        _safe_run(slack_chat.run(session, text), session, client)
    )


async def _on_button(body, client, worker: Worker, approved: bool) -> None:
    action = body["actions"][0]
    session_id: str = action["value"]
    msg_ts: str = body["message"]["ts"]
    channel_id: str = body["channel"]["id"]

    # Replace Block Kit buttons immediately to prevent double-clicks
    status = "Action confirmed. Executing..." if approved else "Action rejected."
    try:
        await client.chat_update(
            channel=channel_id,
            ts=msg_ts,
            text=status,
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": status},
                }
            ],
        )
    except Exception:
        logger.warning("Failed to clear confirmation buttons", exc_info=True)

    session = SlackSession.load(session_id)
    if session is None:
        logger.warning("Button click for unknown session %s, ignoring.", session_id)
        return

    session.awaiting_confirmation = False
    session.is_processing = True
    session.save()

    slack_chat = SlackChat(client, worker, channel_id)
    asyncio.create_task(
        _safe_run(slack_chat.resume_from_confirmation(session, approved), session, client)
    )


async def _safe_run(coro, session: SlackSession, client) -> None:
    """Wrap a background coroutine; on error post to Slack and reset session flags."""
    try:
        await coro
    except Exception as e:
        logger.error("Background task failed", exc_info=True)
        try:
            await client.chat_postMessage(
                channel=session.channel_id,
                thread_ts=session.thread_ts,
                text=f"An error occurred: {e}",
            )
        except Exception:
            logger.error("Failed to post error message to Slack", exc_info=True)
        session.is_processing = False
        session.awaiting_confirmation = False
        session.save()


if __name__ == "__main__":
    main()
