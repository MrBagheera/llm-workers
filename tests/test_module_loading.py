import argparse

from llm_workers_console.chat import chat_with_llm_script
from llm_workers.utils import setup_logging_from_args, add_common_logging_args

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Test for running embedded LLM scripts."
    )
    add_common_logging_args(parser)
    args: argparse.Namespace = parser.parse_args()

    setup_logging_from_args(args)

    chat_with_llm_script("llm_workers:generic-assistant.yaml")
