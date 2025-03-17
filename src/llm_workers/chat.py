import argparse
import logging
import sys
from logging import getLogger
from typing import Optional, Union, Any
from uuid import UUID

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.syntax import Syntax

from llm_workers.api import ConfirmationRequest
from llm_workers.context import StandardContext
from llm_workers.utils import setup_logging, LazyFormatter
from llm_workers.worker import Worker

logger = getLogger(__name__)

class ChatSession:
    def __init__(self, console: Console, script_file: str):
        self._script_file = script_file
        self._console = console
        self._context = StandardContext.from_file(script_file)
        if not self._context.config.chat:
            raise ValueError(f"'chat' section is missing from '{self._script_file}'")
        self._worker = Worker(self._context.config.chat, self._context, top_level=True)
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
        self._chunks_len = 0

    def run(self):
        config = self._context.config.chat
        if config.default_prompt is not None:
            self._pre_input = config.default_prompt

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
                self._chunks_len = 0
                logger.debug("Running new prompt for #%s:\n%r", self._iteration, LazyFormatter(message))
                self._messages.extend(self._worker.invoke(self._messages, config={"callbacks": self._callbacks}))
                self._iteration = self._iteration + 1
        except KeyboardInterrupt:
            self._finished = True
        except EOFError:
            pass

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
            script_file = self._script_file
        elif len(params) == 1:
            script_file = params[0]
        else:
            self._print_help(params)
            return

        self._console.print(f"(Re)loading LLM script from {script_file}", style="bold white")
        self._script_file = script_file
        self._context = StandardContext.from_file(script_file)
        if not self._context.config.chat:
            raise ValueError(f"'chat' section is missing from '{self._script_file}'")
        self._worker = Worker(self._context.config.chat, self._context, top_level=True)

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

    def process_model_chunk(self, token: str):
        self._chunks_len = self._chunks_len + len(token)
        print(token, end="", flush=True)

    def process_model_message(self, message: BaseMessage):
        if self._chunks_len > 0:
            print()
            self._chunks_len = 0
        confidential = getattr(message, 'confidential', False)
        if confidential:
            self._console.print("[Message marked as confidential, not shown to AI Assistant]", style="bold red")
        self._console.print(message.content)
        if confidential:
            self._console.print("[Message marked as confidential, not shown to AI Assistant]", style="bold red")

    def process_tool_start(self, name: str):
        if self._chunks_len > 0:
            print()
            self._chunks_len = 0
        self._console.print(f"Running tool {name}", style="bold white")

    def process_confirmation_request(self, request: ConfirmationRequest):
        self._console.print("\n\n")
        self._console.print(f"AI assistant wants to {request.action}:", style="bold green")
        if len(request.args) == 1:
            arg = request.args[0]
            if arg.format is not None:
                self._console.print(Syntax(arg.value, arg.format))
            else:
                self._console.print(f"{arg.value}:", style="bold white")
        else:
            for arg in request.args:
                self._console.print(f"{arg.name}:")
                if arg.format is not None:
                    self._console.print(Syntax(arg.value, arg.format))
                else:
                    self._console.print(f"{arg.value}:", style="bold white")

        self._console.print("Do you approve?", style="bold green", end="")
        self._console.print(" Answer \"yes\" to proceed, anything else to reject with reason: (Meta+Enter or Escape,Enter to submit)", style="grey69 italic")
        user_input = prompt(multiline=True)
        if user_input.lower() in ["yes", "y"]:
            request.approved = True
        else:
            request.reject_reason = user_input

class ChatSessionCallbackDelegate(BaseCallbackHandler):
    """Delegates selected callbacks to ChatSession"""

    def __init__(self, chat_session: ChatSession):
        self._chat_session = chat_session

    def on_llm_new_token(self, token: str, *, chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        # prefer chunk.text as token is broken for AWS Bedrock
        if chunk is not None and isinstance(chunk, ChatGenerationChunk):
            token = chunk.text
        elif chunk is not None and isinstance(chunk, GenerationChunk):
            token = chunk.text
        if len(token) > 0:
            self._chat_session.process_model_chunk(token)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, *, run_id: UUID,
                      parent_run_id: Optional[UUID] = None, tags: Optional[list[str]] = None,
                      metadata: Optional[dict[str, Any]] = None, inputs: Optional[dict[str, Any]] = None,
                      **kwargs: Any) -> Any:
        # TODO support ui_hint
        self._chat_session.process_tool_start(serialized.get("name", "<tool>"))

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, tags: Optional[list[str]] = None,
                        metadata: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        if name == "on_ai_message":
            self._chat_session.process_model_message(data)
        elif name == "request_confirmation":
            self._chat_session.process_confirmation_request(data)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    args = parser.parse_args()

    load_dotenv()
    _console = Console()
    if args.verbose:
        set_verbose(True)
    if args.debug:
        set_debug(True)
    if args.debug:
        setup_logging(console_level=logging.DEBUG)
    else:
        setup_logging(console_level=logging.WARN)

    chat_session = ChatSession(_console, args.script_file)

    with get_openai_callback() as cb:
        chat_session.run()

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


if __name__ == "__main__":
    main()