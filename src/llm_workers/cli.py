import argparse
import json
import logging
import sys
from typing import Any, Optional

from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import Runnable

from llm_workers.api import UserContext
from llm_workers.workers_context import StandardWorkersContext
from llm_workers.user_context import StandardUserContext
from llm_workers.tools.custom_tool import create_statement_from_model
from llm_workers.utils import setup_logging, prepare_cache, is_cache_prepared


def run_llm_script(
        script_name: str,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        user_context: Optional[UserContext] = None
) -> None:
    """Load LLM script and run it taking input from command line or stdin.

    Args:
        :param script_name: The name of the script to run. Can be either file name or `module_name:resource.yaml`.
        :param parser: Argument parser for command-line arguments.
        :param args: parsed command line arguments to look for `--verbose`, `--debug` and positional arguments.
        :param user_context: custom implementation of UserContext if needed, defaults to StandardUserContext
    """
    if user_context is None:
        user_context = StandardUserContext.load()

    prepare_cache()

    context = StandardWorkersContext.load(script_name, user_context)
    if context.config.cli is None:
        parser.error(f"No CLI configuration found in {script_name}.")

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
                if isinstance(result, str):
                    print(result)
                else:
                    print(json.dumps(result, indent=2))
        else:
            if args.inputs:
                for input in args.inputs:
                    result = worker.invoke({"input": input})
                    if isinstance(result, str):
                        print(result)
                    else:
                        print(json.dumps(result, indent=2))
            else:
                parser.error(f"No inputs provided in {args.script_file}.")

    print(f"Total Tokens: {cb.total_tokens}", file=sys.stderr)
    print(f"Prompt Tokens: {cb.prompt_tokens}", file=sys.stderr)
    print(f"Completion Tokens: {cb.completion_tokens}", file=sys.stderr)
    print(f"Total Cost (USD): ${cb.total_cost}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    # Optional arguments
    parser.add_argument('--verbose', action='count', default=0, help="Enable verbose output. Can be used multiple times to increase verbosity.")
    parser.add_argument('--debug', action='count', default=0, help="Enable debug mode. Can be used multiple times to increase verbosity.")
    # Positional argument for the script file
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    # Optional arguments for prompts or stdin input
    parser.add_argument('inputs', nargs='*', help="Inputs for the script (or use '--' to read from stdin).")
    args = parser.parse_args()

    setup_logging(debug_level = args.debug, verbosity = args.verbose, log_filename = "llm-workers.log")

    run_llm_script(args.script_file, parser, args)


if __name__ == "__main__":
    main()