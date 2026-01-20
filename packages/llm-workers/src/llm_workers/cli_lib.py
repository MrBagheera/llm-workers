"""Library functions for CLI batch processing (without main entry point)."""

import argparse
import json
import sys
from typing import Any, Optional

from llm_workers.api import UserContext, ExtendedRunnable, WorkerNotification
from llm_workers.cache import prepare_cache
from llm_workers.config import CliConfig
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.tools.custom_tool import create_statement_from_model
from llm_workers.user_context import StandardUserContext
from llm_workers.utils import LazyFormatter
from llm_workers.worker_utils import ensure_env_vars_defined, split_result_and_notifications
from llm_workers.workers_context import StandardWorkersContext


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
        user_config = StandardUserContext.load_config()
        environment = EvaluationContext.default_environment()
        ensure_env_vars_defined(environment, user_config.env)
        user_context = StandardUserContext(user_config, environment)

    prepare_cache()

    script = StandardWorkersContext.load_script(script_name)
    ensure_env_vars_defined(user_context.environment, script.env)
    context = StandardWorkersContext(script, user_context)

    cli = context.config.cli
    if cli is None:
        parser.error(f"No CLI configuration found in {script_name}.")

    # Determine the inputs
    inputs: list[Any] = []
    for arg in args.inputs:
        if arg == '-':
            for line in sys.stdin:
                inputs.append(line.strip())
        else:
            inputs.append(arg)
    if cli.process_input == 'all_as_list':
        inputs = [inputs]

    token_tracker: CompositeTokenUsageTracker = context.run(_run, cli, context, user_context, inputs)
    if not token_tracker.is_empty:
        print(token_tracker.format_total_usage(), file=sys.stderr)


def _run(cli: CliConfig, context: StandardWorkersContext, user_context: UserContext, inputs: list[Any]):
    """Execute worker for each input and return token tracker."""
    tools = context.get_tools('cli', cli.tools)
    local_tools = {tool.name: tool for tool in tools}
    worker: ExtendedRunnable[Any] = create_statement_from_model(cli.do, context, local_tools)
    evaluation_context = context.evaluation_context
    token_tracker = CompositeTokenUsageTracker(user_context.models)
    for input in inputs:
        evaluation_context = EvaluationContext({'input': input}, parent=evaluation_context)
        print(f'Processing input: {input}', file=sys.stderr, flush=True)
        generator = worker.yield_notifications_and_result(evaluation_context, token_tracker, config=None)
        while True:
            try:
                chunk = next(generator)
                if not isinstance(chunk, WorkerNotification):
                    raise ValueError(f"Statement yielded non-notification chunk: {LazyFormatter(chunk)}")
                if chunk.text:
                    print(chunk.text, file=sys.stderr, flush=True)
            except StopIteration as e:
                result = e.value
                break
        if not cli.json_output:
            print(str(result))
        elif cli.json_output == 'pretty':
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result, indent=None, ensure_ascii=False))
        print('\n')
        sys.stdout.flush()
    return token_tracker
