"""Command-line entry point for llm-workers-cli."""

import argparse
from llm_workers.cli_lib import run_llm_script
from llm_workers.utils import setup_logging


def main():
    """Main entry point for llm-workers-cli command."""
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    # Optional arguments
    parser.add_argument('--verbose', action='count', default=0, help="Enable verbose output. Can be used multiple times to increase verbosity.")
    parser.add_argument('--debug', action='count', default=0, help="Enable debug mode. Can be used multiple times to increase verbosity.")
    # Positional argument for the script file
    parser.add_argument('script_file', type=str, help="Path to the script file.")
    # Optional arguments for prompts or stdin input
    parser.add_argument('inputs', nargs='*', help="Inputs for the script (or use '-' to read from stdin).")
    args = parser.parse_args()

    setup_logging(debug_level=args.debug, verbosity=args.verbose, log_filename="llm-workers.log")

    run_llm_script(args.script_file, parser, args)


if __name__ == "__main__":
    main()
