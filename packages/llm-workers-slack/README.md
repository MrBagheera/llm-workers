# llm-workers-slack

Slack Socket Mode integration for LLM Workers.

## Overview

`llm-workers-slack` connects an LLM Workers script to a Slack workspace via Socket Mode, enabling threaded conversations with full tool support including interactive confirmation prompts.

## Installation

```bash
pip install llm-workers-slack
```

## Setup

### 1. Create a Slack App

In [api.slack.com/apps](https://api.slack.com/apps):

- **Socket Mode**: enable it and generate an **App-Level Token** (`xapp-...`) with the `connections:write` scope
- **OAuth & Permissions**: add bot scopes `app_mentions:read` and `chat:write`, then install the app to your workspace to get a **Bot User OAuth Token** (`xoxb-...`)
- **Event Subscriptions**: subscribe to the `app_mention` bot event

### 2. Configure credentials

Add to `~/.config/llm-workers/.env`:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 3. Run the bot

```bash
llm-workers-slack my-script.yaml
```

The bot connects via Socket Mode — no public URL or webhook required.

## Usage

### Starting a conversation

@mention the bot in any channel to start a new session:

```
@mybot What files are in the current directory?
```

The bot replies in a thread. Continue the conversation by @mentioning it inside that same thread.

### Confirmation prompts

When the LLM wants to run a tool that requires confirmation (e.g. executing a shell command), the bot posts a message with **Yes** and **No** buttons. Click one to proceed or cancel.

### Session state

Each thread maps to a session file (`{channel_id}__{thread_ts}.json`) stored in the working directory. Set `SLACK_SESSIONS_DIR` to store files elsewhere:

```
SLACK_SESSIONS_DIR=/var/lib/llm-workers/sessions
```

Session files are never deleted automatically — set up an external cron if cleanup is needed.

## Limitations (PoC)

- File uploads and parsing are not supported
- External channel history is not referenced
- Unmentioned replies in threads are ignored

## Documentation

Full documentation: https://mrbagheera.github.io/llm-workers/

## License

See main repository for license information.
