import argparse

from llm_workers_console.chat import chat_with_llm_script
from llm_workers.utils import setup_logging

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Test for running embedded LLM scripts."
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    args: argparse.Namespace = parser.parse_args()

    setup_logging(args)

    chat_with_llm_script("llm_workers:generic-assistant.yaml")
