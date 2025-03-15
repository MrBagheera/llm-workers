import argparse
import logging
import sys
from typing import Any

from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import Runnable

from llm_workers.context import StandardContext
from llm_workers.tools.custom_tool import create_statement_from_model
from llm_workers.utils import setup_logging

load_dotenv()


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
        'inputs', nargs='*', help="Inputs for the script (or use '--' to read from stdin)."
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

    context = StandardContext.from_file(args.script_file)
    if context.config.cli is None:
        parser.error(f"No CLI configuration found in {args.script_file}.")
    worker: Runnable[dict[str, Any], Any]
    try:
        worker = create_statement_from_model(["input"], context.config.cli, context)
    except Exception as e:
        logging.error("Failed to create worker from CLI configuration", exc_info=True)
        parser.error(f"Failed to create worker from CLI configuration: {e}")

    with get_openai_callback() as cb:
        # Determine the input mode
        if '--' in sys.argv:
            if args.inputs:
                parser.error("Cannot use both command-line inputs and '--'.")
            for input in sys.stdin:
                input = input.strip()
                result = worker.invoke({"input": input})
                print(result)
        else:
            if args.inputs:
                for input in args.inputs:
                    result = worker.invoke({"input": input})
                    print(result)
            else:
                parser.error(f"No inputs provided in {args.script_file}.")

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


if __name__ == "__main__":
    main()