import argparse
import sys
from collections.abc import Iterator
from typing import List
import logging

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage

from llm_workers.utils import setup_logging
from llm_workers.worker import LlmWorker

load_dotenv()


def print_model_output(chunks: Iterator[List[BaseMessage]]):
    for chunk in chunks:
        for message in chunk:
            if isinstance(message, AIMessage):
                if message.content != "":
                    print(message.content)


def main():
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

    if args.verbose:
        set_verbose(True)
    if args.debug:
        set_debug(True)

    if args.debug:
        setup_logging(console_level=logging.DEBUG)
    elif args.verbose:
        setup_logging(console_level=logging.INFO)
    else:
        setup_logging(console_level=logging.WARNING)

    worker = LlmWorker(args.script_file)

    with get_openai_callback() as cb:
        try:
            # Determine the input mode
            if '--' in sys.argv:
                if args.prompts:
                    parser.error("Cannot use both command-line prompts and '--'.")
                for line in sys.stdin:
                    line = line.strip()
                    final_prompt = line if worker.default_prompt is None else f"{worker.default_prompt} {line}"
                    print_model_output(worker.stream(final_prompt))
            else:
                if args.prompts:
                    for prompt in args.prompts:
                        final_prompt = prompt if worker.default_prompt is None else f"{worker.default_prompt} {prompt}"
                        print_model_output(worker.stream(final_prompt))
                else:
                    if worker.default_prompt is not None:
                        print_model_output(worker.stream(worker.default_prompt))
                    else:
                        parser.error(f"No prompts provided and no default prompt set in {args.script_file}.")
        finally:
            worker.close()

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


if __name__ == "__main__":
    main()