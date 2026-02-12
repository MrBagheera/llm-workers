"""Command-line entry point for llm-workers-chat."""

import argparse
import sys
from logging import getLogger

from llm_workers.chat_history import ChatHistory
from llm_workers.utils import setup_logging_from_args, add_common_logging_args
from llm_workers_console.chat import chat_with_llm_script

logger = getLogger(__name__)

_default_script_file = "llm_workers:generic-assistant.yaml"


def main():
    """Main entry point for llm-workers-chat command."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for LLM scripts."
    )
    add_common_logging_args(parser)
    parser.add_argument('--resume', action='store_true', help="Resume from last auto-saved session (uses `.last.chat.yaml`)")
    parser.add_argument('script_file', type=str, nargs='?', help="Path to the script file. Generic assistant script will be used if omitted.", default=_default_script_file)
    args = parser.parse_args()

    log_file = setup_logging_from_args(args, log_filename="llm-workers.log")
    print(f"Logging to {log_file}", file=sys.stderr)

    if args.resume:
        try:
            filename = '.last.chat.yaml' if args.script_file == _default_script_file else args.script_file
            chat_history = ChatHistory.load_from_yaml(filename)
        except FileNotFoundError as e:
            parser.error(f"Resume file {e.filename} not found")
            # exits
        except Exception as e:
            logger.error(f"Failed to load resume file", exc_info=True)
            parser.error(f"Failed to load resume file: {e}")
            # exits

        script_file = chat_history.script_name
        history = chat_history.messages
    else:
        script_file = args.script_file
        history = []

    chat_with_llm_script(script_file, history=history)


if __name__ == "__main__":
    main()
