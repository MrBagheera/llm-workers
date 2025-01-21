import sys
from collections.abc import Iterator
from typing import Any
from rich.console import Console
from rich.text import Text

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from llm_workers.worker import build_worker

load_dotenv()
console = Console()


def print_model_output(stream: Iterator[dict[str, Any]]):
    for chunk in stream:
        agent_state = chunk["agent"]
        for message in agent_state["messages"]:
            text = Text()
            if isinstance(message, HumanMessage):
                text.append("Human: ", style="bold blue")
                text.append(message.content)
            elif isinstance(message, ToolMessage):
                text.append("Tool Call: ", style="bold gray")
                text.append(message.content)
            elif isinstance(message, AIMessage):
                text.append("AI: ", style="bold green")
                text.append(message.content)
            else:
                text.append("Unknown: ", style="bold red")
                text.append(message)
            console.print(text)


if __name__ == "__main__":
    # check if config file specified as first parameter
    if len(sys.argv) == 2:
        config_filename = sys.argv[1]
    else:
        print("Usage: python3 main.py <config_file>")
        sys.exit(1)
    worker = build_worker(config_filename)
    print_model_output(worker.stream({"messages": [""]}))
