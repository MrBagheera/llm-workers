import argparse
import logging
import sys
from collections.abc import Iterator
from logging import getLogger
from typing import List

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console

from llm_workers.utils import format_tool
from llm_workers.worker import LlmWorker

logger = getLogger(__name__)

class ChatSession:
    def __init__(self, console: Console, worker: LlmWorker):
        self._console = console
        self._worker = worker
        self._iteration = 1
        self._messages = list[BaseMessage]()
        self._history = InMemoryHistory()

    def run(self):
        if worker.default_prompt is not None:
            console.print(f"#{self._iteration} Prompt:", style="bold green")
            prompt = worker.default_prompt.strip()
            console.print(prompt)
            self._history.append_string(prompt)
            self._process_model_output(worker.stream(prompt))

        session = PromptSession(history = self._history)
        try:
            while True:
                console.print(f"[bold green]#{self._iteration} Your input: [/bold green] [grey69 italic](Meta+Enter or Escape,Enter to submit, /help for commands list)[/grey69 italic]")
                text = session.prompt(multiline=True)
                if text.lower().strip() == "/bye":
                    return
                i = self._messages.copy()
                i.append(text)
                logger.debug(f"Running new prompt for #{self._iteration}:\n{i}")
                self._process_model_output(worker.stream(i))
        except EOFError:
            pass

    def _process_model_output(self, chunks: Iterator[List[BaseMessage]]):
        for chunk in chunks:
            for message in chunk:
                if isinstance(message, HumanMessage):
                    pass
                elif isinstance(message, AIMessage):
                    if message.content != "":
                        console.print(f"#{self._iteration} AI:", style="bold green")
                        console.print(message.content)
                        self._iteration = self._iteration + 1
                    if len(message.tool_calls) > 0:
                        for tool_call in message.tool_calls:
                            console.print(f"Running tool {format_tool(tool_call)}", style="bold white")
                elif isinstance(message, ToolMessage):
                    console.print(f"Tool {message.name} has finished", style="bold white")
                else:
                    console.print(f"#{self._iteration} Unknown({message.type}):", style="bold red")
                    console.print(message.content)
                self._messages.append(message)
                logger.debug(f"Appending {message} to session history")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    args = parser.parse_args()

    load_dotenv()
    worker = LlmWorker(args.script_file)
    if args.verbose:
        set_verbose(True)
    if args.debug:
        set_debug(True)
    if args.verbose or args.debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

    console = Console()
    chat_session = ChatSession(console, worker)
    chat_session.run()
