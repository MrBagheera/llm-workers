import argparse

from llm_workers.chat import build_and_run

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Test for running embedded LLM scripts."
    )
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    args: argparse.Namespace = parser.parse_args()

    build_and_run("llm_workers:coding-assistant.yaml", args)
