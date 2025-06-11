import argparse
import sys
from argparse import Namespace
from logging import getLogger
from typing import Optional, Union, Any
from uuid import UUID

from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from llm_workers.api import ConfirmationRequest, CONFIDENTIAL
from llm_workers.context import StandardContext
from llm_workers.utils import setup_logging, LazyFormatter, find_and_load_dotenv, FileChangeDetector, \
    open_file_in_default_app, is_safe_to_open, prepare_cache, get_key_press
from llm_workers.worker import Worker

logger = getLogger(__name__)


class _ChatSessionContext:
    worker: Worker
    context: StandardContext
    script_name: str

    def __init__(self, script_file: str):
        self.script_name = script_file
        self.context = StandardContext.load(script_file)
        if not self.context.config.chat:
            raise ValueError(f"'chat' section is missing from '{self.script_name}'")
        self.worker = Worker(self.context.config.chat, self.context, top_level=True)
        self.file_monitor = FileChangeDetector(
            path='.',
            included_patterns=self.context.config.chat.file_monitor_include,
            excluded_patterns=self.context.config.chat.file_monitor_exclude)


class ChatSession:
    def __init__(self, console: Console, script_name: str):
        self._console = console
        self._chat_context = _ChatSessionContext(script_name)
        self._file_monitor: Optional[FileChangeDetector] = None
        self._iteration = 1
        self._messages = list[BaseMessage]()
        self._history = InMemoryHistory()
        self.commands = {
            "help": self._print_help,
            "reload": self._reload,
            "rewind": self._rewind,
            "bye": self._bye,
        }
        self._finished = False
        self._pre_input = ""
        self._callbacks = [ChatSessionCallbackDelegate(self)]
        self._streamed_message_id = None
        self._streamed_reasoning_index: Optional[int] = None
        self._has_unfinished_output = False
        self._running_tools_depths = {}

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

        session = PromptSession(history = self._history)
        try:
            while not self._finished:
                if len(self._messages) > 0:
                    print()
                    print()
                    print()
                self._console.print(f"#{self._iteration} Your input:", style="bold green", end="")
                self._console.print(" (Meta+Enter or Escape,Enter to submit, /help for commands list)", style="grey69 italic")
                text = session.prompt(default=self._pre_input.strip(), multiline=True)
                self._pre_input = ""
                if self._parse_and_run_command(text):
                    continue
                # submitting input to the worker
                self._console.print(f"#{self._iteration} AI Assistant:", style="bold green")
                message = HumanMessage(text)
                self._messages.append(message)
                self._streamed_reasoning_index = None
                self._has_unfinished_output = False
                self._streamed_message_id = None
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
        if command in self.commands:
            self.commands[command](params)
        else:
            print(f"Unknown command: {command}.")
            self._print_help([])
        return True

    # noinspection PyUnusedLocal
    def _print_help(self, params: list[str]):
        """-                 Shows this message"""
        print("Available commands:")
        for cmd, func in self.commands.items():
            doc = func.__doc__.strip()
            print(f"  /{cmd} {doc}")

    def _reload(self, params: list[str]):
        """[<script.yaml>] - Reloads given LLM script (defaults to current)"""
        if len(params) == 0:
            script_file = self._chat_context.script_name
        elif len(params) == 1:
            script_file = params[0]
        else:
            self._print_help(params)
            return

        self._console.print(f"(Re)loading LLM script from {script_file}", style="bold white")
        logger.debug(f"Reloading LLM script from {script_file}")
        try:
            self._chat_context = _ChatSessionContext(script_file)
        except Exception as e:
            self._console.print(f"Failed to load LLM script from {script_file}: {e}", style="bold red")
            logger.warning(f"Failed to load LLM script from {script_file}: {e}", exc_info=True)

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

    def process_model_chunk(self, token: str, message: Optional[BaseMessage]):
        self._streamed_message_id = message.id if message is not None else None
        if message is not None and isinstance(message, AIMessage) and self._chat_config.show_reasoning:
            reasoning = self._extract_reasoning(message)
            if len(reasoning) > 0:
                if self._has_unfinished_output:
                    print()
                    self._has_unfinished_output = False
                if self._streamed_reasoning_index is None:
                    self._console.print("Reasoning:", style="bold white")
                    self._streamed_reasoning_index = reasoning[0][0]
                for (index, text) in reasoning:
                    if index != self._streamed_reasoning_index:
                        print()
                    self._streamed_reasoning_index = index
                    self._console.print(text, style="bold white", end = "")
        if len(token) > 0:
            if self._streamed_reasoning_index is not None:
                print()
                self._streamed_reasoning_index = None
            self._has_unfinished_output = True
            print(token, end="", flush=True)

    def process_model_message(self, message: BaseMessage):
        self._messages.append(message)
        if not isinstance(message, AIMessage):
            return
        if self._has_unfinished_output or self._streamed_reasoning_index is not None:
            print()
            self._has_unfinished_output = False
            self._streamed_reasoning_index = None
        if self._streamed_message_id is not None and self._streamed_message_id == message.id:
            if isinstance(message, AIMessage):
                self._streamed_message_id = None
                return
        if self._chat_config.show_reasoning:
            reasoning = self._extract_reasoning(message)
            if len(reasoning) > 0:
                self._console.print("Reasoning:", style="bold white")
                for (index, text) in reasoning:
                    self._console.print(text, style="bold white")
        # text
        confidential = getattr(message, CONFIDENTIAL, False)
        if confidential:
            self._console.print("[Message below is confidential, not shown to AI Assistant]", style="bold red")
        self._console.print(message.text())
        if confidential:
            self._console.print("[Message above is confidential, not shown to AI Assistant]", style="bold red")

    @staticmethod
    def _extract_reasoning(message: AIMessage) -> list[(int, str)]:
        reasoning: list[(int, str)] = []
        if isinstance(message.content, list):
            i = 0
            for block in message.content:
                if isinstance(block, dict):
                    # noinspection PyShadowingBuiltins
                    type = block.get('type', None)
                    if type == 'reasoning_content':
                        reasoning_content = block.get("reasoning_content", {})
                        text = reasoning_content.get("text", None)
                        if text is not None:
                            index = block.get('index', i)
                            reasoning.append((index, str(text)))
                i = i + 1
        return reasoning

    def process_tool_start(self, name: str, inputs: dict[str, Any], run_id: UUID, parent_run_id: Optional[UUID]):
        if self._has_unfinished_output or self._streamed_reasoning_index is not None:
            print()
            self._has_unfinished_output = False
            self._streamed_reasoning_index = None
        message = self._chat_context.context.get_start_tool_message(name, inputs)
        if message is None:
            return
        if parent_run_id is not None and parent_run_id in self._running_tools_depths:
            # increase depth of running tool
            depth = self._running_tools_depths[parent_run_id] + 1
            ident = "  " * depth
            self._running_tools_depths[run_id] = depth
            self._console.print(f"{ident}└ {message}...", style="bold white")
        else:
            self._running_tools_depths[run_id] = 0
            self._console.print(f"⏺ {message}...", style="bold white")

    def process_confirmation_request(self, request: ConfirmationRequest):
        self._console.print("\n\n")
        self._console.print(f"AI assistant wants to {request.action}:", style="bold green")
        if len(request.args) == 1:
            arg = request.args[0]
            if arg.format is not None:
                self._console.print(Syntax(arg.value, arg.format))
            else:
                self._console.print(arg.value, style="bold white")
        else:
            for arg in request.args:
                self._console.print(f"{arg.name}:")
                if arg.format is not None:
                    self._console.print(Syntax(arg.value, arg.format))
                else:
                    self._console.print(arg.value, style="bold white")
        while True:
            self._console.print("Do you approve (y/n)?", style="bold green", end="")
            response = get_key_press()
            print()
            if response == 'y':
                request.approved = True
                return
            elif response == 'n':
                request.approved = False
                return

    def _handle_changed_files(self):
        changes = self._chat_context.file_monitor.check_changes()
        to_open = []
        created = changes.get('created', [])
        if len(created) > 0:
            to_open += created
            self._console.print(f"Files created: {', '.join(created)}", style="bold white")
        modified = changes.get('modified', [])
        if len(modified) > 0:
            to_open += modified
            self._console.print(f"Files updated: {', '.join(modified)}", style="bold white")
        deleted = changes.get('deleted', [])
        if len(deleted) > 0:
            self._console.print(f"Files deleted: {', '.join(deleted)}", style="bold white")
        if not self._chat_config.auto_open_changed_files:
            return
        for filename in to_open:
            if not is_safe_to_open(filename):
                continue
            open_file_in_default_app(filename)


class ChatSessionCallbackDelegate(BaseCallbackHandler):
    """Delegates selected callbacks to ChatSession"""

    def __init__(self, chat_session: ChatSession):
        self._chat_session = chat_session

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
        self._chat_session.process_tool_start(serialized.get("name", "<tool>"), inputs, run_id, parent_run_id)

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, tags: Optional[list[str]] = None,
                        metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        if name == "request_confirmation":
            self._chat_session.process_confirmation_request(data)


def chat_with_llm_script(script_name: str, args: Namespace):
    """
    Load LLM script and chat with it.

    Args:
        script_name: The name of the script to run. Can be either file name or `module_name:resource.yaml`.
        args: command line arguments to look for `--verbose` and `--debug`
    """
    find_and_load_dotenv(".config/llm-workers/.env")
    prepare_cache(create_dir=False)

    # Create a console object for output
    console = Console()

    # Create a chat session and run it
    chat_session = ChatSession(console, script_name)
    with get_openai_callback() as cb:
        chat_session.run()

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='count', default=0, help="Enable verbose output. Can be used multiple times to increase verbosity.")
    parser.add_argument('--debug', action='count', default=0, help="Enable debug mode. Can be used multiple times to increase verbosity.")
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    args = parser.parse_args()

    setup_logging(debug_level = args.debug, verbosity = args.verbose, log_filename = "llm-workers.log")

    chat_with_llm_script(args.script_file, args)


if __name__ == "__main__":
    main()