import argparse
import logging
import sys
from collections.abc import Iterator
from logging import getLogger
from typing import List

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from llm_workers.utils import format_tool_call, setup_logging
from llm_workers.worker import LlmWorker

logger = getLogger(__name__)

class ChatSession:
    def __init__(self, console: Console, script_file: str):
        self._script_file = script_file
        self._console = console
        self._worker = LlmWorker(script_file)
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

    def run(self):
        if self._worker.default_prompt is not None:
            self._pre_input = self._worker.default_prompt

        session = PromptSession(history = self._history)
        try:
            while not self._finished:
                if len(self._messages) > 0:
                    print()
                self._console.print(f"[bold green]#{self._iteration} Your input: [/bold green] [grey69 italic](Meta+Enter or Escape,Enter to submit, /help for commands list)[/grey69 italic]")
                text = session.prompt(default=self._pre_input, multiline=True)
                self._pre_input = ""
                if self._parse_and_run_command(text):
                    continue
                # submitting input to the worker
                i = self._messages.copy()
                i.append(text)
                logger.debug(f"Running new prompt for #{self._iteration}:\n{i}")
                self._process_model_output(self._worker.stream(i))
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
        self._worker.close()
        self._script_file = script_file
        self._worker = LlmWorker(script_file)

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
        logger.debug(f"Rewinding session to {target_iteration}")
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

    def _process_model_output(self, chunks: Iterator[List[BaseMessage]]):
        for chunk in chunks:
            for message in chunk:
                if isinstance(message, HumanMessage):
                    self._iteration = self._iteration + 1
                    pass
                elif isinstance(message, AIMessage):
                    if message.content != "":
                        self._console.print(f"#{self._iteration} AI:", style="bold green")
                        self._console.print(message.content)
                    if len(message.tool_calls) > 0:
                        for tool_call in message.tool_calls:
                            self._console.print(f"Running tool {format_tool_call(tool_call)}", style="bold white")
                elif isinstance(message, ToolMessage):
                    self._console.print(f"Tool {message.name} has finished", style="bold white")
                else:
                    self._console.print(f"#{self._iteration} Unknown({message.type}):", style="bold red")
                    self._console.print(message.content)
                self._messages.append(message)
                logger.debug(f"Appending {repr(message)} to session history")

    def close(self):
        self._worker.close()


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
        try:
            chat_session.run()
        finally:
            chat_session.close()

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


if __name__ == "__main__":
    main()