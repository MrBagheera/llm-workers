import sys
from collections.abc import Iterator
from typing import Any

import asyncio

from langchain.globals import get_debug
from rich.console import Console
from rich.text import Text

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from llm_workers.worker import build_worker

load_dotenv()
console = Console()


async def print_model_output(stream: Iterator[dict[str, Any]]):
    for chunk in stream:
        if get_debug():
            print(chunk)
        agent_state = chunk[next(iter(chunk))] # first key regardless of node
        for message in agent_state["messages"]:
            text = Text()
            if isinstance(message, HumanMessage):
                text.append("Human: ", style="bold blue")
                text.append(message.content)
            elif isinstance(message, ToolMessage):
                text.append(f"Tool Call ({message.name}): ", style="bold white")
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
    asyncio.run(print_model_output(worker.stream({"messages": [""]}, stream_mode="updates")))

