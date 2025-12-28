"""Interactive chat session for LLM scripts (library functions only, no main entry point)."""

import sys
from logging import getLogger
from typing import Optional, Callable

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from llm_workers.api import ConfirmationRequest, ConfirmationResponse, UserContext, WorkerNotification
from llm_workers.cache import prepare_cache
from llm_workers.chat_history import ChatHistory
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.user_context import StandardUserContext
from llm_workers.utils import LazyFormatter, FileChangeDetector, \
    open_file_in_default_app, is_safe_to_open
from llm_workers.worker import Worker
from llm_workers.worker_utils import ensure_env_vars_defined, set_max_start_tool_msg_length
from llm_workers.workers_context import StandardWorkersContext
from llm_workers_console.chat_completer import ChatCompleter
from llm_workers_console.console import ConsoleController

logger = getLogger(__name__)



class _ChatSessionContext:
    worker: Worker
    context: StandardWorkersContext
    script_name: str

    def __init__(self, script_file: str, user_context: UserContext, context: StandardWorkersContext):
        self.script_name = script_file
        self.user_context = user_context
        self.context = context
        if not self.context.config.chat:
            raise ValueError(f"'chat' section is missing from '{self.script_name}'")
        self.worker = Worker(self.context.config.chat, self.context, scope='chat')
        self.file_monitor = FileChangeDetector(
            path='.',
            included_patterns=self.user_context.user_config.display_settings.file_monitor_include,
            excluded_patterns=self.user_context.user_config.display_settings.file_monitor_exclude)


class ChatSession:
    commands: dict[str, Callable[[list[str]], None]]
    commands_config: dict[str, dict]
    alias_to_command: dict[str, str]

    def __init__(self, console: Console):
        self._console = console
        self._iteration = 1
        self._messages = list[BaseMessage]()
        self._history = InMemoryHistory()

        # Structured command configuration with optional aliases and params
        self.commands_config = {
            "help": {
                "function": self._print_help,
                "description": "Shows this message"
            },
            "rewind": {
                "function": self._rewind,
                "description": "Rewinds chat session to input N (default to -1 = previous)",
                "params": "[N]"
            },
            "exit": {
                "aliases": ["bye", "quit"],
                "function": self._exit,
                "description": "Ends chat session"
            },
            "model": {
                "function": self._model,
                "description": "Switch to specified model (fast, default, thinking)",
                "params": "<model>"
            },
            "display": {
                "function": self._display,
                "description": "Show or modify display settings",
                "params": "[<setting> [<value>]]"
            },
            "cost": {
                "aliases": ["tokens", "usage"],
                "function": self._cost,
                "description": "Show token usage for current and total session"
            },
            "export": {
                "function": self._export,
                "description": "Export chat history as <name>.md markdown file",
                "params": "<name>"
            },
            "clear": {
                "aliases": ["new", "reset"],
                "function": self._clear_command,
                "description": "Reset chat session and clear screen"
            },
            "save": {
                "function": self._save,
                "description": "Save chat session to <name>.chat.yaml file",
                "params": "<name>"
            },
        }

        # Build commands dict for backward compatibility
        self.commands = {cmd: config["function"] for cmd, config in self.commands_config.items()}

        # Build alias lookup table
        self.alias_to_command = {}
        for cmd, config in self.commands_config.items():
            # Add the primary command name
            self.alias_to_command[cmd] = cmd
            # Add any aliases
            if "aliases" in config:
                for alias in config["aliases"]:
                    self.alias_to_command[alias] = cmd

        # Initialize command completer
        self._completer = ChatCompleter(self.commands_config)
        self._finished = False
        self._pre_input = ""

    @staticmethod
    def run(console: Console, script_file: str, user_context: UserContext,
            history: Optional[list[BaseMessage]] = None) -> CompositeTokenUsageTracker:
        """Runs chat session in async loop."""
        # Changing from sync to async API is WIP - it is forced on us because of MCP server.
        # So we start async loop here using calling thread,
        # and then run actual chat session synchronously using separate thread.
        chat_session = ChatSession(console)
        script = StandardWorkersContext.load_script(script_file)
        ensure_env_vars_defined(user_context.environment, script.env)
        workers_context = StandardWorkersContext(script, user_context)
        workers_context.run(chat_session._run_chat_loop, script_file, user_context, workers_context, history or [])
        return chat_session._token_tracker

    def _run_chat_loop(self, script_file: str, user_context: UserContext, workers_context: StandardWorkersContext, history: list[BaseMessage]):
        self._token_tracker = CompositeTokenUsageTracker(user_context.models)
        self._chat_context = _ChatSessionContext(script_file, user_context, workers_context)
        self._console_controller = ConsoleController(self._console, self._display_settings)

        # Display user banner if configured
        if self._chat_config.user_banner is not None:
            self._console.print(Markdown(self._chat_config.user_banner))
            self._console.print()

        # Load and replay session if resuming
        if history:
            self._replay_session(history)
        else:
            if self._chat_config.default_prompt is not None:
                self._pre_input = self._chat_config.default_prompt

        session = PromptSession(history=self._history, completer=self._completer, style=self._completer.style)
        try:
            while not self._finished:
                if len(self._messages) > 0:
                    print()
                    print()
                    print()

                # Display token usage before prompting for input
                if self._iteration > 1 and self._display_settings.show_token_usage:  # Only show after first response and if enabled
                    usage_display = self._token_tracker.format_current_usage()
                    if usage_display is not None:
                        self._console.print(usage_display)

                self._console.print(f"#{self._iteration} Your input:", style="bold green", end="")
                self._console.print(f" (Model: {self._chat_context.worker.model_ref}, Meta+Enter or Escape,Enter to submit, /help for commands list)", style="grey69 italic")
                text = session.prompt(default=self._pre_input.strip(), multiline=True)
                self._pre_input = ""
                if self._parse_and_run_command(text):
                    continue
                # submitting input to the worker
                self._console.print(f"#{self._iteration} Assistant:", style="bold green")
                message = HumanMessage(text)
                self._messages.append(message)
                self._console_controller.clear()
                self._chat_context.file_monitor.check_changes() # reset
                set_max_start_tool_msg_length(self._console.width - 20)
                logger.debug("Running new prompt for #%s:\n%r", self._iteration, LazyFormatter(message))
                try:
                    confirmation_response: Optional[ConfirmationResponse] = None
                    while True:
                        messages: list[BaseMessage | ConfirmationResponse] = self._messages if not confirmation_response \
                            else self._messages + [confirmation_response]
                        confirmation_response = None
                        for message in self._chat_context.worker.stream(messages, stream = True):
                            item = message[0]
                            if isinstance(item, WorkerNotification):
                                self._process_notification(item)
                            elif isinstance(item, ConfirmationRequest):
                                confirmation_response = self._process_confirmation_request(item)
                            else:
                                self._process_model_message(item)
                        if confirmation_response is None:
                            break
                except Exception as e:
                    self._console_controller.clear()
                    logger.error(f"Error: {e}", exc_info=True)
                    self._console.print(f"Unexpected error in worker: {e}", style="bold red")
                    self._console.print("If subsequent conversation fails, try rewinding to previous message", style="bold red")
                self._handle_changed_files()
                self._iteration = self._iteration + 1
        except KeyboardInterrupt:
            self._finished = True
        except EOFError:
            self._finished = True

    @property
    def _display_settings(self):
        return self._chat_context.user_context.user_config.display_settings

    @property
    def _user_context(self) -> UserContext:
        return self._chat_context.user_context

    @property
    def _chat_config(self):
        return self._chat_context.context.config.chat

    @property
    def _available_models(self):
        return [model.name for model in self._user_context.models]

    def _parse_and_run_command(self, message: str) -> bool:
        message = message.strip()
        if len(message) == 0:
            return False
        if message[0] != "/":
            return False
        message = message[1:].split()
        command = message[0]
        params = message[1:]

        # Resolve alias to primary command
        if command in self.alias_to_command:
            primary_command = self.alias_to_command[command]
            self.commands[primary_command](params)
        else:
            print(f"Unknown command: {command}.")
            self._print_help([])
        return True

    # noinspection PyUnusedLocal
    def _print_help(self, params: list[str]):
        """-                 Shows this message"""
        print("Available commands:")

        # Use completer's formatting for consistency
        for cmd, config in self.commands_config.items():
            _, aligned_display = self._completer._format_command_display(cmd, config)
            print(aligned_display)

    def _rewind(self, params: list[str]):
        """[N] - Rewinds chat session to input N (default to previous)"""
        if len(params) == 0:
            target_iteration = -1
        elif len(params) == 1:
            try:
                target_iteration = int(params[0])
            except ValueError:
                self._print_help(params)
                return
        else:
            self._print_help(params)
            return
        if target_iteration < 0:
            target_iteration = max(self._iteration + target_iteration, 1)
        else:
            target_iteration = min(self._iteration, target_iteration)
        if target_iteration == self._iteration:
            return
        logger.info(f"Rewinding session to #{target_iteration}")
        self._console.clear()
        self._iteration = target_iteration

        i = 0
        iteration = 1
        while i < len(self._messages):
            message = self._messages[i]
            if isinstance(message, HumanMessage):
                if iteration == target_iteration:
                    # truncate history
                    self._messages = self._messages[:i]
                    self._iteration = target_iteration
                    self._pre_input = str(message.content)
                    return
                iteration = iteration + 1
            i = i + 1

    # noinspection PyUnusedLocal
    def _exit(self, params: list[str]):
        """- Ends chat session"""
        # Auto-save to .last.chat.yaml if messages exist
        if len(self._messages) > 0:
            try:
                filename = self._save_session_to_file('.last.chat.yaml')
                logger.info(f"Auto-saved session to {filename}")
            except Exception as e:
                # Log but don't interrupt exit
                logger.warning("Failed to auto-save session on exit", exc_info=True)

        self._finished = True

    # noinspection PyUnusedLocal
    def _clear_command(self, params: list[str]):
        """- Reset chat session and clear screen"""
        # Clear the console screen
        self._console.clear()

        # Reset message history
        self._messages.clear()

        # Reset iteration counter
        self._iteration = 1

        # Clear any pre-populated input
        self._pre_input = ""

    def _model(self, params: list[str]):
        """<model> - Switch to specified model (fast, default, thinking)"""
        if len(params) != 1:
            self._console.print("Usage: /model <model_name>")
            self._console.print(f"Available models: {', '.join(self._available_models)}")
            return

        model_name = params[0]
        if model_name not in self._available_models:
            self._console.print(f"Unknown model: {model_name}", style="bold red")
            self._console.print(f"Available models: {', '.join(self._available_models)}")
            return

        if model_name == self._chat_context.worker.model_ref:
            self._console.print(f"Already using model: {model_name}",)
            return

        try:
            # Use the Worker's model_ref setter to switch models
            self._chat_context.worker.model_ref = model_name
            self._console.print(f"Switched to model: {model_name}")
        except Exception as e:
            self._console.print(f"Failed to switch to model {model_name}: {e}", style="bold red")
            logger.warning(f"Failed to switch to model {model_name}", exc_info=True)

    def _get_boolean_settings(self) -> dict[str, bool]:
        """Get all boolean display settings as a dictionary."""
        settings = self._display_settings
        return {
            "show_token_usage": settings.show_token_usage,
            "show_reasoning": settings.show_reasoning,
            "auto_open_changed_files": settings.auto_open_changed_files,
            "markdown_output": settings.markdown_output
        }

    def _display(self, params: list[str]):
        """[<setting> [<value>]] - Show or modify display settings"""
        if len(params) == 0:
            # Show all current boolean settings
            settings = self._get_boolean_settings()
            self._console.print("Current display settings:")
            for setting, value in settings.items():
                self._console.print(f"  {setting}: {value}")
            return

        if len(params) == 1:
            # Show specific setting
            setting_name = params[0]
            settings = self._get_boolean_settings()
            if setting_name not in settings:
                self._console.print(f"Unknown setting: {setting_name}", style="bold red")
                self._console.print(f"Available settings: {', '.join(settings.keys())}")
                return

            value = settings[setting_name]
            self._console.print(f"{setting_name}: {value}")
            return

        if len(params) == 2:
            # Set specific setting
            setting_name = params[0]
            new_value_str = params[1].lower()

            settings = self._get_boolean_settings()
            if setting_name not in settings:
                self._console.print(f"Unknown setting: {setting_name}", style="bold red")
                self._console.print(f"Available settings: {', '.join(settings.keys())}")
                return

            # Parse boolean value
            if new_value_str in ['true', '1', 'on', 'yes']:
                new_value = True
            elif new_value_str in ['false', '0', 'off', 'no']:
                new_value = False
            else:
                self._console.print(f"Invalid value: {params[1]}", style="bold red")
                self._console.print("Valid values: true, false, 1, 0, on, off, yes, no")
                return

            # Set the value
            display_settings = self._display_settings
            setattr(display_settings, setting_name, new_value)

            status = "enabled" if new_value else "disabled"
            self._console.print(f"{setting_name.replace('_', ' ').title()} {status}", style="bold green")
            return

        # Too many parameters
        self._console.print("Usage: /display [<setting> [<value>]]", style="bold red")

    # noinspection PyUnusedLocal
    def _cost(self, params: list[str]):
        """- Show token usage"""
        total_usage = self._token_tracker.format_total_usage()
        if total_usage is not None:
            self._console.print(total_usage)
        else:
            self._console.print("  No tokens used in this session\n")

    def _export(self, params: list[str]):
        """<name> - Export chat history as <name>.md markdown file"""
        if len(params) != 1:
            self._console.print("Usage: /export <filename>", style="bold red")
            return

        filename = params[0]
        if not filename.endswith('.md'):
            filename += '.md'

        try:
            markdown_content = self._generate_markdown_export()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            self._console.print(f"Chat history exported to {filename}")
        except Exception as e:
            self._console.print(f"Failed to export chat history: {e}", style="bold red")
            logger.warning(f"Failed to export chat history to {filename}", exc_info=True)

    def _save(self, params: list[str]):
        """<name> - Save chat session to <name>.chat.yaml file"""
        if len(params) != 1:
            self._console.print("Usage: /save <filename>", style="bold red")
            return

        filename = params[0]
        if not filename.endswith('.chat.yaml'):
            filename += '.chat.yaml'

        if len(self._messages) == 0:
            self._console.print("No messages to save", style="bold yellow")
            return

        try:
            filename = self._save_session_to_file(filename)
            self._console.print(f"Session saved to {filename}", style="bold green")
        except Exception as e:
            self._console.print(f"Failed to save session: {e}", style="bold red")
            logger.warning(f"Failed to save session to {filename}", exc_info=True)

    def _generate_markdown_export(self) -> str:
        """Generate markdown content from chat history"""
        if not self._messages:
            return "# Chat History\n\nNo messages to export.\n"

        markdown_lines = []
        current_iteration = 0
        last_ai_iteration = 0

        for i, message in enumerate(self._messages):
            # Skip tool messages (tool call responses)
            if isinstance(message, ToolMessage):
                continue

            # Add separator between messages (except before the first message)
            if len(markdown_lines) > 1:
                markdown_lines.append("---\n")

            if isinstance(message, HumanMessage):
                current_iteration += 1
                markdown_lines.append(f"# User #{current_iteration}\n")
                markdown_lines.append(f"{message.content}\n")

            elif isinstance(message, AIMessage):
                if current_iteration != last_ai_iteration:
                    markdown_lines.append(f"# Assistant #{current_iteration}\n")
                    last_ai_iteration = current_iteration

                # Add message text content
                if message.content:
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, dict):
                                type = block.get('type', None)
                                if type == 'text':
                                    text = block.get("text", "")
                                    markdown_lines.append(f"{text}\n")
                            else:
                                markdown_lines.append(f"{str(block)}\n")
                    else:
                        markdown_lines.append(f"{message.content}\n")

                # Add tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown_tool')
                        tool_args = tool_call.get('args', {})
                        markdown_lines.append(f"Calling `{tool_name}`:\n")
                        markdown_lines.append("```\n")
                        result = None
                        if len(tool_args) == 1:
                            # single argument, format as is
                            arg_value = next(iter(tool_args.values()))
                            if isinstance(arg_value, str):
                                result = f"{arg_value}\n"
                        if result is None:
                            formatted_args = LazyFormatter(tool_args).__repr__()
                            result = f"{formatted_args}\n"
                        markdown_lines.append(result)
                        markdown_lines.append("```\n")
                        markdown_lines.append("\n")

        return "\n".join(markdown_lines)

    def _save_session_to_file(self, filename: str) -> str:
        """Save session to a YAML file."""
        history = ChatHistory(script_name= self._chat_context.script_name, messages = self._messages)
        return history.save_to_yaml(filename)

    def _replay_messages(self, messages: list[BaseMessage]) -> int:
        """Replay messages to console using existing display code, returns iteration to continue from."""
        iteration = 0
        last_human_message = False

        for i, message in enumerate(messages):
            # Skip ToolMessages in display
            if isinstance(message, ToolMessage):
                continue

            if isinstance(message, HumanMessage):
                # Display user input
                if not last_human_message:
                    iteration += 1
                    last_human_message = True
                    self._console.print()
                    self._console.print(f"#{iteration} Your input:", style="bold green")
                self._console.print(message.content)
                self._console.print()

            elif isinstance(message, AIMessage):
                # Display AI response
                if last_human_message:
                    last_human_message = False
                    self._console.print(f"#{iteration} Assistant:", style="bold green")

                # reuse existing display logic
                self._console_controller.process_model_message(message)

        return iteration + 1

    def _replay_session(self, history: list[BaseMessage]) -> None:
        """Replays all messages to console."""
        self._iteration = self._replay_messages(history)
        self._messages = history.copy()

    def _process_notification(self, notification: WorkerNotification):
        """Process a WorkerNotification based on its type."""
        if notification.type == 'thinking_start':
            self._console_controller.show_thinking()
        elif notification.type == 'thinking_end':
            self._console_controller.clear_thinking_message()
        elif notification.type == 'ai_output_chunk':
            if notification.text:
                self._console_controller.process_output_chunk(message_id=notification.message_id, text=notification.text)
        elif notification.type == 'ai_reasoning_chunk':
            if notification.text:
                self._console_controller.process_reasoning_chunk(message_id=notification.message_id, text=notification.text, index=notification.index)
        elif notification.type == 'tool_start':
            if notification.text and notification.run_id:
                self._console_controller.process_tool_start_notification(notification.text, notification.run_id, notification.parent_run_id)
        elif notification.type == 'tool_end':
            # No action needed for tool_end currently
            pass

    def _process_model_message(self, message: BaseMessage):
        self._messages.append(message)

        # Update token tracking with automatic model detection
        default_model = self._chat_context.worker.model_ref
        self._token_tracker.update_from_message(message, default_model)

        if not isinstance(message, AIMessage):
            return
        self._console_controller.process_model_message(message)

    def _process_confirmation_request(self, request: ConfirmationRequest) -> ConfirmationResponse:
        self._console_controller.clear()
        approved_tool_calls: list[str] = []

        # Iterate through all tool calls and ask for confirmation independently
        for tool_call_id, tool_request in request.tool_calls.items():
            self._console.print(f"\nAI assistant wants to {tool_request.action}:", style="bold green")

            if len(tool_request.params) == 1:
                arg = tool_request.params[0]
                if arg.format is not None:
                    self._console.print(Syntax(arg.value, arg.format))
                else:
                    self._console.print(arg.value)
            else:
                for arg in tool_request.params:
                    self._console.print(f"{arg.name}:")
                    if arg.format is not None:
                        self._console.print(Syntax(arg.value, arg.format))
                    else:
                        self._console.print(arg.value)

            while True:
                response = self._console.input("[bold green]Do you approve (y/n)?[/bold green] ").strip().lower()
                if response in ['y', 'yes']:
                    approved_tool_calls.append(tool_call_id)
                    break
                elif response in ['n', 'no']:
                    break
                else:
                    self._console.print("Please enter 'y' or 'n'", style="bold red")

        return ConfirmationResponse(approved_tool_calls=approved_tool_calls)

    def get_token_usage_summary(self) -> str | None:
        """Get formatted token usage summary."""
        return self._token_tracker.format_current_usage()

    def _handle_changed_files(self):
        changes = self._chat_context.file_monitor.check_changes()
        to_open = []
        created = changes.get('created', [])
        if len(created) > 0:
            to_open += created
            self._console.print(f"Files created: {', '.join(created)}")
        modified = changes.get('modified', [])
        if len(modified) > 0:
            to_open += modified
            self._console.print(f"Files updated: {', '.join(modified)}")
        deleted = changes.get('deleted', [])
        if len(deleted) > 0:
            self._console.print(f"Files deleted: {', '.join(deleted)}")
        if not self._display_settings.auto_open_changed_files:
            return
        for filename in to_open:
            if not is_safe_to_open(filename):
                continue
            open_file_in_default_app(filename)


def chat_with_llm_script(script_name: str, user_context: Optional[UserContext] = None,
                         history: Optional[list[BaseMessage]] = None):
    """
    Load LLM script and chat with it.

    Args:
        :param script_name: The name of the script to run. Can be either file name or `module_name:resource.yaml`.
        :param user_context: custom implementation of UserContext if needed, defaults to StandardUserContext
        :param history: Optional message history to resume from.
    """
    if user_context is None:
        user_config = StandardUserContext.load_config()
        environment = EvaluationContext.default_environment()
        ensure_env_vars_defined(environment, user_config.env)
        user_context = StandardUserContext(user_config, environment)

    prepare_cache()

    console = Console()

    tokens_counts = ChatSession.run(console, script_name, user_context, history)

    # Print detailed per-model session token summary
    if user_context.user_config.display_settings.show_token_usage:
        session_summary = tokens_counts.format_total_usage()
        if session_summary is not None:
            print(f"{session_summary}", file=sys.stderr)
