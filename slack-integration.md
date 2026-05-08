# Slack Bolt Integration Specification (PoC)

**Target:** AI Coding Agent
**Objective:** Implement a Slack Socket Mode integration for an existing CLI AI chatbot.

## 1. Architecture & Infrastructure
* **Framework:** Slack Bolt.
* **Connection:** Socket Mode (requires App-Level Token and Bot User OAuth Token).
* **Scopes:** `app_mentions:read`, `chat:write`.
* **Unsupported Features (PoC):** File uploads/parsing, referencing external channel history, or listening to unmentioned thread replies.

## 2. State Management & Session IDs
State is maintained via local JSON files. Do not implement file cleanup logic (handled by external cron).

* **Session ID Format:** `{channel_id}__{thread_ts_normalized}`
    * Extract `channel` and `ts` (or `thread_ts` if threaded) from the Slack payload.
    * Replace the `.` in the timestamp with `_`.
* **Storage:** One JSON file per session, named `{session_id}.json`.
* **Read/Write:** Load the file into memory on a valid event, append the new state/message, and save.
* **State Flags:** * `is_processing` (boolean): Prevents concurrent message handling.
    * `awaiting_confirmation` (boolean): Halts text processing until a UI button is clicked.

## 3. Event Routing & Concurrency
Subscribe to the `app_mention` event for text input and `block_actions` for button clicks.

### Text Input (`app_mention`)
* **Top-Level Mention (No `thread_ts` in payload):**
    * Action: Initialize a new chat session. Treat the message text (stripped of the bot mention) as the first user input.
* **Threaded Mention (Payload contains `thread_ts`):**
    * Action: Query the local session state using the `session_id`.
    * **Case A (No session exists):** Respond in the thread: *"Please start a new chat session by @-mentioning me in a top-level message."* Drop event.
    * **Case B (Session exists):** Proceed to Concurrency Check.

### Concurrency Check
When evaluating a valid `app_mention` against an existing session:
* If `is_processing == true`: Respond in thread: *"Already thinking on a previous message, ignoring this."* Drop event.
* If `awaiting_confirmation == true`: Respond in thread: *"Please answer the pending confirmation above before continuing."* Drop event.
* If both are `false`: Set `is_processing = true`, save state, and begin AI processing.

## 4. Execution Flow: Standard Chat
When a valid text message passes routing and concurrency checks:

1. **Acknowledge:** Send a `chat.postMessage` to the thread containing the text: *"Thinking..."*.
2. **Capture:** Store the `ts` of this "Thinking..." message.
3. **Process:** Pass the conversation history to the existing AI logic.
4. **Log Update:** Once tool execution finishes, use `chat.update` to replace the *"Thinking..."* message text with the AI tool execution logs.
    * Format as Markdown code blocks.
    * Truncate the log string to 39,000 characters to prevent Slack API rejection.
5. **Final Response:** Send the final AI output as a new `chat.postMessage` to the thread.
6. **Unlock:** Append the AI's response to the session JSON, set `is_processing = false`, and save.

## 5. Execution Flow: Interactive Confirmation
When the AI logic requires user approval (e.g., executing a destructive command):

### Requesting Confirmation
1. **Send UI:** Use `chat.postMessage` to send a Block Kit payload to the thread.
    * Include a `section` block with the markdown text (max 3,000 chars).
    * Include an `actions` block with "Yes" and "No" buttons.
    * **Crucial:** Embed the current `session_id` into the `value` attribute of both buttons.
2. **Lock State:** Update the session JSON: set `is_processing = false` and `awaiting_confirmation = true`. Save state.

### Handling the Button Click (`block_actions`)
1. **Acknowledge:** Immediately call `ack()` on the incoming event to satisfy Slack's 3-second timeout.
2. **Extract:** Retrieve the `session_id` from the button's `value` attribute.
3. **Clear UI:** Use `chat.update` on the message `ts` to remove the Block Kit buttons (preventing double-clicks) and replace them with standard text (e.g., *"Action confirmed. Executing..."*).
4. **Resume State:** Load the session JSON, set `awaiting_confirmation = false` and `is_processing = true`.
5. **Execute:** Pass the button choice back to the AI logic, proceed with standard execution, and follow the **Standard Chat** flow for the final response.