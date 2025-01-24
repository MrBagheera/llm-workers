import argparse
import sys
from collections.abc import Iterator
from typing import List

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, HumanMessage
from prompt_toolkit.history import InMemoryHistory
from requests import session
from rich.console import Console
from rich.text import Text
from prompt_toolkit import prompt, PromptSession

from llm_workers.worker import LlmWorker

load_dotenv()
console = Console()


def print_model_output(chunks: Iterator[List[BaseMessage]]):
    for chunk in chunks:
        for message in chunk:
            if isinstance(message, HumanMessage):
                pass
            elif isinstance(message, AIMessage):
                if message.content != "":
                    console.print("AI:", style="bold green")
                    console.print(message.content)
                if len(message.tool_calls) > 0:
                    for tool_call in message.tool_calls:
                        console.print(f"Running tool {tool_call["name"]}", style="bold white")
            elif isinstance(message, ToolMessage):
                console.print(f"Tool {message.name} has finished", style="bold white")
            else:
                console.print(f"Unknown({message.type}):", style="bold red")
                console.print(message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    args = parser.parse_args()

    worker = LlmWorker(args.script_file)
    if args.verbose:
        set_verbose(True)
    if args.debug:
        set_debug(True)

    history = InMemoryHistory()
    if worker.default_prompt is not None:
        console.print("Prompt:", style="bold green")
        prompt = worker.default_prompt.strip()
        console.print(prompt)
        history.append_string(prompt)
        print_model_output(worker.stream(worker.default_prompt))

    session = PromptSession(history = history)
    try:
        while True:
            console.print("[bold green]Your input: [/bold green] [grey69 italic](press Meta+Enter or Escape followed by Enter to submit, submit \'bye\' or press ^D to exit)[/grey69 italic]")
            text = session.prompt(multiline=True)
            if text.lower() == "bye":
                break
            print_model_output(worker.stream(text))
    except EOFError:
        pass
