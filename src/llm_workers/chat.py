import argparse
import sys
from logging import getLogger
from typing import Optional, Union, Any, Dict, List, Tuple, Callable
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from llm_workers.api import ConfirmationRequest, CONFIDENTIAL, UserContext
from llm_workers.chat_completer import ChatCompleter
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.user_context import StandardUserContext
from llm_workers.utils import setup_logging, LazyFormatter, FileChangeDetector, \
    open_file_in_default_app, is_safe_to_open, prepare_cache
from llm_workers.worker import Worker
from llm_workers.workers_context import StandardWorkersContext

logger = getLogger(__name__)



class _ChatSessionContext:
    worker: Worker
    context: StandardWorkersContext
    script_name: str

    def __init__(self, script_file: str, user_context: UserContext):
        self.script_name = script_file
        self.user_context = user_context
        self.context = StandardWorkersContext.load(script_file, user_context)
        if not self.context.config.chat:
            raise ValueError(f"'chat' section is missing from '{self.script_name}'")
        self.worker = Worker(self.context.config.chat, self.context, top_level=True)
        self.file_monitor = FileChangeDetector(
            path='.',
            included_patterns=self.user_context.user_config.display_settings.file_monitor_include,
            excluded_patterns=self.user_context.user_config.display_settings.file_monitor_exclude)


class ChatSession:
    commands: dict[str, Callable[[list[str]], None]]
    commands_config: dict[str, dict]
    alias_to_command: dict[str, str]

    def __init__(self, console: Console, script_name: str, user_context: UserContext):
        self._console = console
        self._user_context = user_context
        self._chat_context = _ChatSessionContext(script_name, user_context)
        self._file_monitor: Optional[FileChangeDetector] = None
        self._iteration = 1
        self._messages = list[BaseMessage]()
        self._history = InMemoryHistory()

        # Structured command configuration with optional aliases and params
        self.commands_config = {
            "help": {
                "function": self._print_help,
                "description": "Shows this message"
            },
            "reload": {
                "function": self._reload,
                "description": "Reloads given LLM script (defaults to current)",
                "params": "[<script.yaml>]"
            },
            "rewind": {
                "function": self._rewind,
                "description": "Rewinds chat session to input N (default to -1 = previous)",
                "params": "[N]"
            },
            "bye": {
                "aliases": ["exit", "quit"],
                "function": self._bye,
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
            "export": {
                "function": self._export,
                "description": "Export chat history as <name>.md markdown file",
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
        self._callbacks = [ChatSessionCallbackDelegate(self)]
        self._token_tracker = CompositeTokenUsageTracker()
        self._streamed_message_id = None
        self._streamed_reasoning_index: Optional[int] = None
        self._has_unfinished_output = False
        self._running_tools_depths = {}
        self._available_models = [model.name for model in self._user_context.models]
        self._thinking_live = None

    @property
    def _chat_config(self):
        return self._chat_context.context.config.chat

    def run(self):
        # Display user banner if configured
        if self._chat_config.user_banner is not None:
            self._console.print(Markdown(self._chat_config.user_banner))
            self._console.print()

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
                if self._iteration > 1 and self._user_context.user_config.display_settings.show_token_usage:  # Only show after first response and if enabled
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
                self._streamed_reasoning_index = None
                self._has_unfinished_output = False
                self._streamed_message_id = None
                self._running_tools_depths.clear()
                self._chat_context.file_monitor.check_changes() # reset
                logger.debug("Running new prompt for #%s:\n%r", self._iteration, LazyFormatter(message))
                try:
                    for message in self._chat_context.worker.stream(self._messages, stream=True, config={"callbacks": self._callbacks}):
                        self.process_model_message(message[0])
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    self._console.print(f"Unexpected error in worker: {e}", style="bold red")
                    self._console.print(f"If subsequent conversation fails, try rewinding to previous message", style="bold red")
                self._handle_changed_files()
                self._iteration = self._iteration + 1
        except KeyboardInterrupt:
            self._finished = True
        except EOFError:
            self._finished = True

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

    def _reload(self, params: list[str]):
        """[<script.yaml>] - Reloads given LLM script (defaults to current)"""
        if len(params) == 0:
            script_file = self._chat_context.script_name
        elif len(params) == 1:
            script_file = params[0]
        else:
            self._print_help(params)
            return

        self._console.print(f"(Re)loading LLM script from {script_file}")
        logger.debug(f"Reloading LLM script from {script_file}")
        try:
            self._chat_context = _ChatSessionContext(script_file, self._user_context)
        except Exception as e:
            self._console.print(f"Failed to load LLM script from {script_file}: {e}", style="bold red")
            logger.warning(f"Failed to load LLM script from {script_file}", exc_info=True)

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
    def _bye(self, params: list[str]):
        """- Ends chat session"""
        self._finished = True

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
        settings = self._user_context.user_config.display_settings
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
            display_settings = self._user_context.user_config.display_settings
            setattr(display_settings, setting_name, new_value)

            status = "enabled" if new_value else "disabled"
            self._console.print(f"{setting_name.replace('_', ' ').title()} {status}", style="bold green")
            return

        # Too many parameters
        self._console.print("Usage: /display [<setting> [<value>]]", style="bold red")

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

    def process_model_chunk(self, token: str, message: Optional[BaseMessage]):
        self._streamed_message_id = message.id if message is not None else None
        if message is not None and isinstance(message, AIMessage) and self._user_context.user_config.display_settings.show_reasoning:
            self.clear_thinking_message()
            reasoning = self._extract_reasoning(message)
            if len(reasoning) > 0:
                if self._has_unfinished_output:
                    print()
                    self._has_unfinished_output = False
                if self._streamed_reasoning_index is None:
                    self._console.print("Reasoning:")
                    self._streamed_reasoning_index = reasoning[0][0]
                for (index, text) in reasoning:
                    if index != self._streamed_reasoning_index:
                        print()
                    self._streamed_reasoning_index = index
                    self._console.print(text, end = "")
        if len(token) > 0:
            self.clear_thinking_message()
            if self._streamed_reasoning_index is not None:
                print()
                self._streamed_reasoning_index = None
            self._has_unfinished_output = True
            print(token, end="", flush=True)

    def process_model_message(self, message: BaseMessage):
        self._messages.append(message)

        # Update token tracking with automatic model detection
        default_model = self._chat_context.worker.model_ref
        self._token_tracker.update_from_message(message, default_model)

        if not isinstance(message, AIMessage):
            return
        self.clear_thinking_message()
        if self._has_unfinished_output or self._streamed_reasoning_index is not None:
            print()
            self._has_unfinished_output = False
            self._streamed_reasoning_index = None
        if self._streamed_message_id is not None and self._streamed_message_id == message.id:
            if isinstance(message, AIMessage):
                self._streamed_message_id = None
                return
        if self._user_context.user_config.display_settings.show_reasoning:
            reasoning = self._extract_reasoning(message)
            if len(reasoning) > 0:
                self._console.print("Reasoning:")
                for (index, text) in reasoning:
                    self._console.print(text)
        # text
        confidential = getattr(message, CONFIDENTIAL, False)
        if confidential:
            self._console.print("[Message below is confidential, not shown to AI Assistant]", style="bold red")
        if self._user_context.user_config.display_settings.markdown_output:
            self._console.print(Markdown(message.text()))
        else:
            self._console.print(message.text())
        if confidential:
            self._console.print("[Message above is confidential, not shown to AI Assistant]", style="bold red")

    @staticmethod
    def _extract_reasoning(message: AIMessage) -> List[Tuple[int, str]]:
        reasoning: List[Tuple[int, str]] = []
        if isinstance(message.content, list):
            i = 0
            for block in message.content:
                if isinstance(block, dict):
                    # noinspection PyShadowingBuiltins
                    type = block.get('type', None)
                    text = None
                    if type == 'reasoning_content':
                        reasoning_content = block.get("reasoning_content", {})
                        text = reasoning_content.get("text", None)
                    elif type == 'reasoning':
                        if 'summary' in block: # OpenAI GPT-5
                            block = block.get("summary")
                            if isinstance(block, list) and len(block) > 0:
                                block = block[0]
                        if 'text' in block:
                            text = block.get("text")
                    if text is not None:
                        index = block.get('index', i)
                        reasoning.append((index, str(text)))
                i = i + 1
        return reasoning

    def process_tool_start(self, name: str, tool_meta: Dict[str, Any], inputs: dict[str, Any], run_id: UUID, parent_run_id: Optional[UUID]):
        self.clear_thinking_message()
        if self._has_unfinished_output or self._streamed_reasoning_index is not None:
            print()
            self._has_unfinished_output = False
            self._streamed_reasoning_index = None
        message = self._chat_context.context.get_start_tool_message(name, tool_meta, inputs)
        if parent_run_id is not None and parent_run_id in self._running_tools_depths:
            # increase depth of running tool
            depth = self._running_tools_depths[parent_run_id] + (1 if message is not None else 0)
            self._running_tools_depths[run_id] = depth
            if message is not None:
                ident = "  " * depth
                self._console.print(f"{ident}└ {message}...")
        else:
            if message is None:
                return  # do not store depth to avoid showing nesting for sub-tools
            self._running_tools_depths[run_id] = 0
            self._console.print(f"⏺ {message}...")

    def process_confirmation_request(self, request: ConfirmationRequest):
        self.clear_thinking_message()
        self._console.print("\n\n")
        self._console.print(f"AI assistant wants to {request.action}:", style="bold green")
        if len(request.args) == 1:
            arg = request.args[0]
            if arg.format is not None:
                self._console.print(Syntax(arg.value, arg.format))
            else:
                self._console.print(arg.value)
        else:
            for arg in request.args:
                self._console.print(f"{arg.name}:")
                if arg.format is not None:
                    self._console.print(Syntax(arg.value, arg.format))
                else:
                    self._console.print(arg.value)
        while True:
            response = self._console.input("[bold green]Do you approve (y/n)?[/bold green] ").strip().lower()
            if response in ['y', 'yes']:
                request.approved = True
                return
            elif response in ['n', 'no']:
                request.approved = False
                return
            else:
                self._console.print("Please enter 'y' or 'n'", style="bold red")

    def get_token_usage_summary(self) -> str | None:
        """Get formatted token usage summary."""
        return self._token_tracker.format_current_usage()

    def get_session_token_summary(self) -> str | None:
        """Get detailed per-model session token summary for exit display."""
        return self._token_tracker.format_total_usage()

    def show_thinking(self):
        """Display 'Thinking...' message using Rich Live display."""
        if not self._thinking_live:
            self._thinking_live = self._console.status("Thinking...", spinner="dots")
            self._thinking_live.start()

    def clear_thinking_message(self):
        """Clear the 'Thinking...' message."""
        if self._thinking_live:
            self._thinking_live.stop()
            self._thinking_live = None

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
        if not self._user_context.user_config.display_settings.auto_open_changed_files:
            return
        for filename in to_open:
            if not is_safe_to_open(filename):
                continue
            open_file_in_default_app(filename)


class ChatSessionCallbackDelegate(BaseCallbackHandler):
    """Delegates selected callbacks to ChatSession"""

    def __init__(self, chat_session: ChatSession):
        self._chat_session = chat_session

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID,
                      parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None,
                      metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        self._chat_session.show_thinking()

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        # prefer chunk.text as token is broken for AWS Bedrock
        if chunk is not None:
            self._chat_session.process_model_chunk(chunk.text, chunk.message)
        else:
            self._chat_session.process_model_chunk(token, message=None)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None,
                      metadata: Optional[dict[str, Any]] = None, inputs: Optional[dict[str, Any]] = None,
                      **kwargs: Any) -> Any:
        self._chat_session.process_tool_start(serialized['name'], metadata, inputs, run_id, parent_run_id)

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, tags: Optional[list[str]] = None,
                        metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        if name == "request_confirmation":
            self._chat_session.process_confirmation_request(data)


def chat_with_llm_script(script_name: str, user_context: Optional[UserContext] = None):
    """
    Load LLM script and chat with it.

    Args:
        :param script_name: The name of the script to run. Can be either file name or `module_name:resource.yaml`.
        :param user_context: custom implementation of UserContext if needed, defaults to StandardUserContext
    """
    if user_context is None:
        user_context = StandardUserContext.load()

    prepare_cache()

    console = Console()

    chat_session = ChatSession(console, script_name, user_context)
    chat_session.run()

    # Print detailed per-model session token summary
    if user_context.user_config.display_settings.show_token_usage:
        session_summary = chat_session.get_session_token_summary()
        if session_summary is not None:
            print(f"{session_summary}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='count', default=0, help="Enable verbose output. Can be used multiple times to increase verbosity.")
    parser.add_argument('--debug', action='count', default=0, help="Enable debug mode. Can be used multiple times to increase verbosity.")
    parser.add_argument('script_file', type=str, nargs='?', help="Path to the script file. Generic assistant script will be used if omitted.", default="llm_workers:generic-assistant.yaml")
    args = parser.parse_args()

    log_file = setup_logging(debug_level = args.debug, verbosity = args.verbose, log_filename = "llm-workers.log")
    print(f"Logging to {log_file}", file=sys.stderr)

    chat_with_llm_script(args.script_file)


if __name__ == "__main__":
    main()