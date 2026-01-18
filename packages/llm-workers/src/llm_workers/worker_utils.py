import logging
import uuid
from typing import Any, Generator
from typing import List, Optional, Dict

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException

from llm_workers.api import WorkerNotification, ExtendedBaseTool, ExtendedExecutionTool
from llm_workers.config import ToolDefinition, EnvVarDefinition
from llm_workers.expressions import StringExpression, EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import matches_patterns, ensure_environment_variable, LazyFormatter

logger =  logging.getLogger(__name__)


def ensure_env_vars_defined(environment: Dict[str, str], env_definitions: Dict[str, EnvVarDefinition]) -> None:
    """
    Process environment variable definitions, prompting user for missing values.

    Args:
        environment: Dictionary representing the current environment
        env_definitions: Dictionary mapping var names to EnvVarDefinition objects

    Note:
        - For persistent=True vars: uses ensure_environment_variable() to prompt and save
        - For persistent=False vars: only prompts if not already set, doesn't save to .env
        - For is_secret=True vars: hides input using prompt_toolkit (if available and TTY)
        - If env var is already set in os.environ, skips prompting
    """
    if not env_definitions:
        return

    for var_name, env_def in env_definitions.items():
        # ensure_environment_variable will check if already set and skip if so
        ensure_environment_variable(
            environment,
            var_name,
            env_def.description,
            is_persistent=env_def.persistent,
            is_secret=env_def.secret
        )


MAX_START_TOOL_MSG_LENGTH = 80

def set_max_start_tool_msg_length(length: int) -> None:
    """
    Set the global maximum length for start tool notification.

    Args:
        length: Maximum length in characters
    """
    global MAX_START_TOOL_MSG_LENGTH
    MAX_START_TOOL_MSG_LENGTH = length

def get_start_tool_message(
        tool_name: str,
        tool_meta: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        ui_hint_override: Optional[StringExpression] = None,
        evaluation_context: Optional[EvaluationContext] = None
) -> str | None:
    # Priority 0: Override from call statement
    if ui_hint_override is not None:
        ctx = evaluation_context if evaluation_context else EvaluationContext(inputs)
        hint = ui_hint_override.evaluate(ctx).strip()
        if hint:
            return hint

    if tool_meta:
        try:
            ui_hint = None
            ui_hint_args = []
            if 'tool_definition' in tool_meta:
                tool_def = tool_meta['tool_definition']
                ui_hint = tool_def.ui_hint
                ui_hint_args = tool_def.ui_hint_args

            # Priority 1a: ui_hint is False
            if ui_hint is False:
                return None
            # Priority 1b: ui_hint is non-empty StringExpression
            elif isinstance(ui_hint, StringExpression):
                hint = ui_hint.evaluate(EvaluationContext(inputs)).strip()
                if hint:
                    return hint

            # Priority 2: Tool-specific hint from ExtendedBaseTool
            if '__extension' in tool_meta:
                extension: ExtendedBaseTool = tool_meta['__extension']
                hint = extension.get_ui_hint(inputs).strip()
                if hint:
                    return hint

            # Priority 3: Explicit True → message with args
            if ui_hint:
                prefix = f"Calling {tool_name}"
                max_args_length = MAX_START_TOOL_MSG_LENGTH - len(prefix) - 2  # account for parentheses
                args_str = format_tool_args(inputs, ui_hint_args, max_args_length)
                return f"{prefix}({args_str})" if args_str else prefix

        except Exception:
            logger.warning(f"Unexpected exception formatting start message for tool {tool_name}", exc_info=True)

    # default
    return f"Running tool {tool_name}"

def format_tool_args(inputs: Dict[str, Any], arg_patterns: List[str], max_length: int) -> str:
    """
    Format tool arguments for UI display, filtering by patterns.

    Args:
        inputs: Dictionary of tool input arguments
        arg_patterns: List of patterns to match argument names (supports negation with !)
        max_length: Maximum length of the formatted string before truncation

    Returns:
        Formatted argument string truncated to max_length with [...] if needed
    """
    if not inputs or not arg_patterns:
        return ""

    result = ""
    result_len = 0
    result_truncated = False
    for key, value in inputs.items():
        if not matches_patterns(key, arg_patterns):
            continue

        key_str = str(key)
        value_str = repr(value)
        # [, ]'key': value
        arg_len = len(key_str) + 4 + len(repr(value)) + (0 if result_len == 0 else 2)
        if result_len + arg_len > max_length:
            result_truncated = True
            # we can't fit this argument, but continue for other args
        else:
            if result_len > 0:
                result += ", "
            result += f"'{key_str}': {value_str}"
            result_len += arg_len
    if result_truncated:
        if result_len > 0:
            result += ", "
        result += "[...]"

    return result

# noinspection PyUnreachableCode
def validate_tool_results(logging_id: str, tool_results: Generator[WorkerNotification, None, Any]) -> Generator[WorkerNotification, None, Any]:
    """
    Yields WorkerNotification objects from the stream.
    Returns the tool result (or None if stream ends before).
    Logs warning if non-notification chunk occurs in stream.
    """
    while True:
        try:
            chunk = next(tool_results)
            if not isinstance(chunk, WorkerNotification):
                logger.warning("%s produced multiple results, skipping %r", logging_id, LazyFormatter(chunk))
            yield chunk
        except StopIteration as e:
            return e.value

def extract_tool_results(logging_id: str, tool_results: Generator[WorkerNotification, None, Any]) -> Generator[WorkerNotification, None, Any]:
    """
    Returns tools results, throws away all notifications.
    Logs warning if non-notification chunk occur.
    """
    while True:
        try:
            chunk = next(tool_results)
            if not isinstance(chunk, WorkerNotification):
                logger.warning("%s produced multiple results, skipping %r", logging_id, LazyFormatter(chunk))
        except StopIteration as e:
            return e.value

def call_tool(
        tool: BaseTool,
        input: dict[str, Any],
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        kwargs: dict[str, Any],
        ui_hint_override: Optional[StringExpression] = None
) -> Generator[WorkerNotification, None, Any]:
    run_id = config.get("run_id", None) if config is not None else None
    child_config = config

    tool_start_text = get_start_tool_message(tool.name, tool.metadata, input, ui_hint_override, evaluation_context)
    if tool_start_text:
        parent_run_id = run_id
        run_id = uuid.uuid4()
        child_config = config.copy() if config is not None else RunnableConfig()
        child_config['run_id'] = run_id
        yield WorkerNotification.tool_start(tool_start_text, run_id, parent_run_id)

    result = None
    try:
        if isinstance(tool, ExtendedExecutionTool):
            result = yield from validate_tool_results(tool.name, tool.run_with_notifications(input, evaluation_context, token_tracker, child_config))
        else:
            result = tool.invoke(input, child_config, **kwargs)
    except ToolException as e:
        logger.warning("Failed to call tool %s", tool.name, exc_info=True)
        result = f"Tool Error: {e}"

    if tool_start_text:
        yield WorkerNotification.tool_end(run_id)

    return result


def split_result_and_notifications(generator: Generator[WorkerNotification, None, Any]) -> tuple[Any, list[WorkerNotification]]:
    notifications: list[WorkerNotification] = []
    while True:
        try:
            chunk = next(generator)
            if not isinstance(chunk, WorkerNotification):
                raise ValueError(f"Statement yielded non-notification chunk: {LazyFormatter(chunk)}")
            notifications.append(chunk)
        except StopIteration as e:
            return e.value, notifications
