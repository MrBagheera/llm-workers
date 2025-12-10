import argparse
import sys
from typing import Optional

from rich.console import Console

from llm_workers.core.api import UserContext
from llm_workers.core.user_context import StandardUserContext
from llm_workers.core.utils import setup_logging, prepare_cache, ensure_env_vars_defined
from llm_workers.console_chat.chat_session import ChatSession


def chat_with_llm_script(script_name: str, user_context: Optional[UserContext] = None):
    """
    Load LLM script and chat with it.

    Args:
        :param script_name: The name of the script to run. Can be either file name or `module_name:resource.yaml`.
        :param user_context: custom implementation of UserContext if needed, defaults to StandardUserContext
    """
    if user_context is None:
        user_config = StandardUserContext.load_config()
        ensure_env_vars_defined(user_config.env)
        user_context = StandardUserContext(user_config)

    prepare_cache()

    console = Console()

    tokens_counts = ChatSession.run(console, script_name, user_context)

    # Print detailed per-model session token summary
    if user_context.user_config.display_settings.show_token_usage:
        session_summary = tokens_counts.format_total_usage()
        if session_summary is not None:
            print(f"{session_summary}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run LLM scripts with prompts from command-line or stdin."
    )
    parser.add_argument('--verbose', action='count', default=0, help="Enable verbose output. Can be used multiple times to increase verbosity.")
    parser.add_argument('--debug', action='count', default=0, help="Enable debug mode. Can be used multiple times to increase verbosity.")
    parser.add_argument('script_file', type=str, nargs='?', help="Path to the script file. Generic assistant script will be used if omitted.", default="llm_workers.core.resources:generic-assistant.yaml")
    args = parser.parse_args()

    log_file = setup_logging(debug_level = args.debug, verbosity = args.verbose, log_filename = "llm-workers.log")
    print(f"Logging to {log_file}", file=sys.stderr)

    chat_with_llm_script(args.script_file)


if __name__ == "__main__":
    main()
