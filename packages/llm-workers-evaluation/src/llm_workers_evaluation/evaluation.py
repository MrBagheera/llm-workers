"""Core evaluation logic for running evaluation suites against LLM scripts."""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel

from llm_workers.api import UserContext, WorkerNotification, WorkersContext
from llm_workers.cache import prepare_cache
from llm_workers.config import ToolsDefinitionStatement, Json
from llm_workers.expressions import (
    EvaluationContext as VarEvaluationContext,
    JsonExpression
)
from llm_workers.token_tracking import CompositeTokenUsageTracker, UsageReport
from llm_workers.tools.custom_tool import create_statement_from_model
from llm_workers.user_context import StandardUserContext
from llm_workers.utils import LazyFormatter, load_yaml
from llm_workers.worker_utils import ensure_env_vars_defined
from llm_workers.workers_context import StandardWorkersContext
from llm_workers_evaluation.config import (
    EvaluationSuiteFile,
    EvaluationTestConfig,
)
from llm_workers_evaluation.tools import TEST_ITERATION_KEY, TEST_LOGS_KEY, log_tool
from llm_workers_evaluation.tools import f_beta_score_tool
from llm_workers_evaluation.tools import linear_score_tool

logger = logging.getLogger(__name__)


class TestResult(BaseModel):
    """Result of running a single test across multiple iterations."""
    average_score: float = 0.0
    scores: Dict[int, float] = {}
    errors: Optional[Dict[int, str]] = None
    logs: Optional[Dict[int, List[Json]]] = None


class EvaluationResults(BaseModel):
    """Complete evaluation results across all suites."""
    final_score: float = 0.0
    usage: Optional[UsageReport] = None
    tests: Dict[str, TestResult] = {}


def validate_score(result: Any) -> float:
    """Validate and clamp result to [0.0, 1.0].

    Args:
        result: The result to validate

    Returns:
        A float clamped to [0.0, 1.0]
    """
    if result is None:
        logger.warning("Test returned None, treating as score 0.0")
        return 0.0

    try:
        score = float(result)
    except (TypeError, ValueError):
        logger.warning(f"Test returned non-numeric value '{result}', treating as score 0.0")
        return 0.0

    if score < 0.0:
        logger.warning(f"Test returned score {score} < 0.0, clamping to 0.0")
        return 0.0

    if score > 1.0:
        logger.warning(f"Test returned score {score} > 1.0, clamping to 1.0")
        return 1.0

    return score


def build_evaluation_context(data: Dict[str, JsonExpression], parent: VarEvaluationContext) -> VarEvaluationContext:
    resolved_data = { key: expr.evaluate(parent) for key, expr in data.items() }
    return VarEvaluationContext(resolved_data, parent=parent)


def merge_tools(
    base: List[ToolsDefinitionStatement],
    override: List[ToolsDefinitionStatement]
) -> List[ToolsDefinitionStatement]:
    """Concatenate tool lists."""
    return list(base) + list(override)


class EvaluationTest:

    def __init__(self,
            test_name: str,
            test_config: EvaluationTestConfig,
            suite_evaluation_context: VarEvaluationContext,
            parent_tools: List[ToolsDefinitionStatement],
            context: WorkersContext
    ):
        self.name = test_name

        merged_tools = merge_tools(test_config.tools, parent_tools)
        tools = context.get_tools('evaluation', merged_tools)
        local_tools = {tool.name: tool for tool in tools}
        self._worker = create_statement_from_model(test_config.do, context, local_tools)

        self._evaluation_context = build_evaluation_context(test_config.data, parent=suite_evaluation_context)

    def _run(self, token_tracker: CompositeTokenUsageTracker, iteration: int, logs_container: List[Json]) -> float:
        """Execute one test iteration and return score.

        Args:
            token_tracker: Token usage tracker
            logs_container: Container for logs generated during this iteration

        Returns:
            Score for this iteration [0.0, 1.0]
        """
        local_evaluation_context = VarEvaluationContext(
            variables={
                TEST_ITERATION_KEY: iteration,
                TEST_LOGS_KEY: logs_container
            },
            parent=self._evaluation_context)
        generator = self._worker.yield_notifications_and_result(local_evaluation_context, token_tracker, config=None)
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

        return validate_score(result)

    def run(
            self,
            token_tracker: CompositeTokenUsageTracker,
            iterations: int,
    ) -> TestResult:
        """Run test N times and return aggregated result.

        Args:
            token_tracker: Token usage tracker
            iterations: Number of iterations to run

        Returns:
            TestResult with scores and average
        """
        result = TestResult()
        result.errors = {}
        result.logs = {}

        for i in range(iterations):
            result.logs[i] = []

            try:
                logger.info(f"Running test '{self.name}' iteration {i + 1}/{iterations}")
                score = self._run(token_tracker, i, result.logs[i])
                result.scores[i] = score
                logger.info(f"Test '{self.name}' iteration {i + 1} score: {score}")
            except Exception as e:
                logger.error(f"Test '{self.name}' iteration {i + 1} failed with error: {e}", exc_info=True)
                result.errors[i] = repr(e)
                result.scores[i] = 0.0

        # Calculate average score
        if result.scores:
            result.average_score = sum(result.scores.values()) / len(result.scores)
        else:
            result.average_score = 0.0

        # Clean up empty optional fields
        if not result.errors:
            result.errors = None
        if result.logs and all(len(log_list) == 0 for log_list in result.logs.values()):
            result.logs = None

        return result


def run_evaluation(
    script_name: str,
    suite_file: str,
    iterations: Optional[int] = None,
    user_context: Optional[UserContext] = None,
) -> EvaluationResults:
    """Main entry point for running evaluations.

    Args:
        script_name: Path to the LLM script file
        suite_file: Path to the evaluation suite YAML file
        iterations: Override for number of iterations (or use suite file default)
        user_context: Custom user context (defaults to StandardUserContext)

    Returns:
        Tuple of (EvaluationResults, CompositeTokenUsageTracker)
    """
    # Initialize user context
    if user_context is None:
        user_config = StandardUserContext.load_config()
        environment = VarEvaluationContext.default_environment()
        ensure_env_vars_defined(environment, user_config.env)
        user_context = StandardUserContext(user_config, environment)

    prepare_cache()

    # Load LLM script
    script = StandardWorkersContext.load_script(script_name)
    ensure_env_vars_defined(user_context.environment, script.env)
    context = StandardWorkersContext(script, user_context)

    # Load evaluation suite
    suite_yaml = load_yaml(suite_file)
    suite_config = EvaluationSuiteFile(**suite_yaml)

    # Use command-line iterations if provided, otherwise use suite file default
    actual_iterations = iterations if iterations is not None else suite_config.iterations

    # Run evaluation within context
    return context.run(
        _run_evaluation_inner,
        suite_config, context, user_context, actual_iterations
    )


def _run_evaluation_inner(
    evaluation_config: EvaluationSuiteFile,
    context: StandardWorkersContext,
    user_context: UserContext,
    iterations: int,
) -> EvaluationResults:
    """Inner evaluation function that runs within the context.

    Args:
        evaluation_config: The evaluation suite configuration
        context: The workers context
        user_context: The user context
        iterations: Number of iterations per test

    Returns:
        Tuple of (EvaluationResults, CompositeTokenUsageTracker)
    """
    # Create results first so LogTool can reference it
    results = EvaluationResults()
    token_tracker = CompositeTokenUsageTracker(user_context.models)

    # FIXME Fugly hack
    shared_tools = context.shared_tools
    shared_tools[f_beta_score_tool.name] = f_beta_score_tool
    shared_tools[linear_score_tool.name] = linear_score_tool
    shared_tools[log_tool.name] = log_tool

    shared_evaluation_context = build_evaluation_context(evaluation_config.shared.data, parent=context.evaluation_context)
    tests = [
        EvaluationTest(
            test_name,
            test_config,
            shared_evaluation_context,
            evaluation_config.shared.tools,
            context
        )
        for test_name, test_config in evaluation_config.tests.items()
    ]

    for test in tests:
        logger.info(f"Running evaluation test '{test.name}'")
        test_result = test.run(token_tracker, iterations)
        results.tests[test.name] = test_result
        logger.info(f"Test '{test.name}' final score: {test_result.average_score}")

    # Calculate overall score from suite scores
    tests_scores = [test.average_score for test in results.tests.values()]
    results.final_score = sum(tests_scores) / len(tests_scores) if tests_scores else 0.0

    # Build usage report
    results.usage = token_tracker.build_usage_report()

    return results


def format_results(results: EvaluationResults) -> str:
    """Format evaluation results as YAML.

    Args:
        results: The evaluation results

    Returns:
        YAML-formatted string
    """
    output = results.model_dump(mode='json', by_alias=True, exclude_none=True)

    return yaml.dump(output, default_flow_style=False, sort_keys=False, allow_unicode=True)
