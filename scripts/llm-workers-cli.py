import argparse
import sys
from collections.abc import Iterator
from typing import List
import logging

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_core.messages import AIMessage, BaseMessage

from llm_workers.worker import LlmWorker

load_dotenv()


def print_model_output(chunks: Iterator[List[BaseMessage]]):
    for chunk in chunks:
        for message in chunk:
            if isinstance(message, AIMessage):
                if message.content != "":
                    print(message.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    # Optional arguments
    parser.add_argument(
        '--verbose', action='store_true', help="Enable verbose output."
    )
    parser.add_argument(
        '--debug', action='store_true', help="Enable debug mode."
    )
    # Positional argument for the script file
    parser.add_argument(
        'script_file', type=str, help="Path to the script file."
    )
    # Optional arguments for prompts or stdin input
    parser.add_argument(
        'prompts', nargs='*', help="Prompts for the script (or use '--' to read from stdin)."
    )
    args = parser.parse_args()

    worker = LlmWorker(args.script_file)
    if args.verbose:
        set_verbose(True)
    if args.debug:
        set_debug(True)
    if args.verbose or args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Determine the input mode
    if '--' in sys.argv:
        if args.prompts:
            parser.error("Cannot use both command-line prompts and '--'.")
        for line in sys.stdin:
            print_model_output(worker.stream(line.strip()))
    else:
        if args.prompts:
            for prompt in args.prompts:
                print_model_output(worker.stream(prompt))
        else:
            if worker.default_prompt is not None:
                print_model_output(worker.stream(worker.default_prompt))
            else:
                parser.error(f"No prompts provided and no default prompt set in {args.script_file}.")
