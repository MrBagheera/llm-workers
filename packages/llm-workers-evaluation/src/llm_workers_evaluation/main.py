"""Command-line entry point for llm-workers-evaluate."""

import argparse

from llm_workers.utils import setup_logging_from_args, add_common_logging_args

from llm_workers_evaluation import run_evaluation, format_results, EvaluationResults


def main():
    """Main entry point for llm-workers-evaluate command."""
    parser = argparse.ArgumentParser(
        description="Run evaluation suites against LLM scripts and report scores."
    )
    # Optional arguments
    add_common_logging_args(parser)
    parser.add_argument(
        '--iterations', '-n', type=int, default=None,
        help="Number of iterations per test (overrides suite file default)."
    )
    # Positional arguments
    parser.add_argument(
        'script_file', type=str,
        help="Path to the LLM script file (or module:resource.yaml)."
    )
    parser.add_argument(
        'evaluation_suite', type=str,
        help="Path to the evaluation suite YAML file."
    )
    args = parser.parse_args()

    setup_logging_from_args(
        args,
        log_filename="llm-workers-evaluate.log"
    )

    results: EvaluationResults = run_evaluation(
        args.script_file,
        args.evaluation_suite,
        iterations=args.iterations
    )

    # Output YAML results to stdout (includes usage stats)
    print(format_results(results))


if __name__ == "__main__":
    main()
