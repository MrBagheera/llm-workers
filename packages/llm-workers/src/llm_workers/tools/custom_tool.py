import logging
import queue
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from logging import Logger
from typing import Type, Any, Optional, Dict, TypeAlias, List, Generator, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, create_model

from llm_workers.api import WorkersContext, WorkerNotification, ExtendedRunnable, ExtendedExecutionTool
from llm_workers.config import Json, CustomToolParamsDefinition, \
    CallDefinition, EvalDefinition, StatementDefinition, IfDefinition, StarlarkDefinition, ForEachDefinition, CustomToolDefinition
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import LazyFormatter, parse_standard_type, TRACE
from llm_workers.worker_utils import call_tool

_default_logger = Logger(__name__)

Statement: TypeAlias = ExtendedRunnable[Json]


class EvalStatement(ExtendedRunnable[Json]):
    def __init__(self, model: EvalDefinition, logger: Logger = _default_logger):
        self._eval_expr = model.eval
        self._store_as = model.store_as
        self._logger = logger

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        result = self._eval_expr.evaluate(evaluation_context)
        if False:  # To make this function return generator, yield statement must exists in it's body
            # noinspection PyUnreachableCode
            yield WorkerNotification()
        if self._store_as:
            evaluation_context.add(self._store_as, result)
        return result # cannot use return here due to generator


# noinspection PyTypeHints
class CallStatement(ExtendedRunnable[Json]):

    def __init__(self, model: CallDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger):
        self._tool = context.get_tool(model.call, local_tools)
        self._params_expr = model.params
        if isinstance(model.catch, list):
            self._catch = model.catch
        elif isinstance(model.catch, str):
            self._catch = [model.catch]
        else:
            self._catch = None
        self._store_as = model.store_as
        self._ui_hint = model.ui_hint
        self._logger = logger

    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        # Evaluate params expression
        target_params = self._params_expr.evaluate(evaluation_context) if self._params_expr else {}
        self._logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params, trim=not self._logger.isEnabledFor(TRACE)))
        try:
            result = yield from call_tool(self._tool, target_params, evaluation_context, token_tracker, config, kwargs, ui_hint_override=self._ui_hint)
            self._logger.debug("Calling tool %s resulted:\n%r", self._tool.name, LazyFormatter(result, trim=not self._logger.isEnabledFor(TRACE)))
            if self._store_as:
                evaluation_context.add(self._store_as, result)
            return result
        except BaseException as e:
            self._logger.debug("Calling tool %s failed: %r", self._tool.name, LazyFormatter(e, trim=False))
            raise self._convert_error(e)

    def _convert_error(self, e: BaseException) -> BaseException:
        if self._catch:
            exception_type = type(e).__name__
            for catch in self._catch:
                if catch == '*' or catch == 'all' or exception_type == catch:
                    return ToolException(str(e), e)
        return e


class FlowStatement(ExtendedRunnable[Json]):

    def __init__(self, model: list[StatementDefinition], context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger):
        self._statements: List[Statement] = []
        for statement_model in model:
            statement = create_statement_from_model(statement_model, context, local_tools, logger)
            self._statements.append(statement)
        self._logger = logger

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        # if we are inside another FlowStatement, inherit its '_' variable
        result = evaluation_context.get('_') # returns None if not defined
        i = 0
        for statement in self._statements:
            inner_context = EvaluationContext({"_": result}, parent=evaluation_context, mutable=False)
            result = yield from statement.yield_notifications_and_result(inner_context, token_tracker, config)
            i += 1
        return result


# noinspection PyTypeHints
class IfStatement(ExtendedRunnable[Json]):
    """
    Executes conditional logic based on boolean expression evaluation.

    If the condition evaluates to a truthy value, executes the 'then' branch.
    Otherwise, executes the 'else' branch (if provided) or returns None.
    """

    def __init__(self, model: IfDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger):
        self._condition_expr = model.if_
        self._then_statement = create_statement_from_model(model.then, context, local_tools, logger)
        self._else_statement = None
        if model.else_ is not None:
            self._else_statement = create_statement_from_model(model.else_, context, local_tools, logger)
        self._store_as = model.store_as
        self._logger = logger

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Generator[WorkerNotification, None, Json]:
        # Evaluate the condition
        condition_result = self._condition_expr.evaluate(evaluation_context)

        # Use Python truthiness
        if condition_result:
            result = yield from self._then_statement.yield_notifications_and_result(
                evaluation_context, token_tracker, config
            )
        elif self._else_statement is not None:
            result = yield from self._else_statement.yield_notifications_and_result(
                evaluation_context, token_tracker, config
            )
        else:
            result = None

        # Store result if requested
        if self._store_as:
            evaluation_context.add(self._store_as, result)

        return result


class StarlarkStatement(ExtendedRunnable[Json]):
    """
    Executes a Starlark script with access to tools and variables.

    Tools from the custom tool's 'tools' field are available as callable functions.
    Input variables and evaluation context are available as global variables.
    Result is returned via 'result' variable or 'run()' function.
    """

    def __init__(self, model: StarlarkDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger):
        self._script = model.starlark
        self._store_as = model.store_as
        # combine local and shared tools (local take precedence)
        self._tools = context.shared_tools | local_tools
        self._logger = logger

        # Compile Starlark script once during initialization
        from llm_workers.starlark import StarlarkExec
        self._executor = StarlarkExec(self._script)

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Generator[WorkerNotification, None, Json]:
        # Extract all variables from evaluation context (including parents)
        global_vars = evaluation_context.extract_all_variables()

        # Create wrapper functions for tools
        # For MVP: execute tools synchronously and collect results
        from llm_workers.worker_utils import split_result_and_notifications

        global_funcs = {}
        for tool_name, tool in self._tools.items():
            # Use closure to capture the correct tool reference
            def create_tool_wrapper(t):
                def wrapper(**params):
                    # Execute tool and collect result synchronously
                    result_gen = call_tool(t, params, evaluation_context, token_tracker, config, kwargs)
                    result, _ = split_result_and_notifications(result_gen)
                    # Note: we lose notifications for MVP, but that's acceptable
                    return result
                return wrapper

            global_funcs[tool_name] = create_tool_wrapper(tool)

        # Execute Starlark script
        result = self._executor.run(global_vars, global_funcs, evaluation_context.logger)

        # Store result if requested
        if self._store_as:
            evaluation_context.add(self._store_as, result)

        # Dummy yield to make this a generator
        if False:
            # noinspection PyUnreachableCode
            yield WorkerNotification()

        return result


@dataclass
class _CompletionSentinel:
    """Signals all tasks are complete."""
    pass


@dataclass
class _ExceptionSentinel:
    """Signals an exception occurred in a worker task."""
    exception: BaseException
    index: Any  # Index/key where the exception occurred


@dataclass
class _IndexedNotification:
    """A notification with its source index for ordering."""
    index: Any
    notification: WorkerNotification


@dataclass
class _IndexedResult:
    """A result with its source index for ordering."""
    index: int
    result: Json


class ForEachStatement(ExtendedRunnable[Json]):
    """
    Iterates over a collection and executes the body for each element.

    - Dict: Maps over values (preserving keys), returns dict
    - Iterable (except str): Maps over elements, returns list of results
    - Other (including str): Treats as single item, returns single result

    Available in body: `_` = current element, `key` = key (for dicts)

    Parallelism:
    - parallelism <= 1: Sequential execution (default)
    - parallelism > 1: Parallel execution with N worker threads
    """

    def __init__(self, model: ForEachDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger):
        self._collection_expr = model.for_each
        self._body_statement = create_statement_from_model(model.do, context, local_tools, logger)
        self._store_as = model.store_as
        self._parallelism = model.parallelism
        self._logger = logger

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Generator[WorkerNotification, None, Json]:
        collection = self._collection_expr.evaluate(evaluation_context)

        # Handle scalars early (non-collection values including strings)
        if not isinstance(collection, (dict, Iterable)) or isinstance(collection, str):
            inner_context = EvaluationContext({"_": collection}, parent=evaluation_context, mutable=False)
            result = yield from self._body_statement.yield_notifications_and_result(
                inner_context, token_tracker, config
            )
            if self._store_as:
                evaluation_context.add(self._store_as, result)
            return result

        # Step 1: Prepare iteration based on collection type
        if isinstance(collection, dict):
            items = list(collection.items())
            make_context = lambda e: EvaluationContext({"_": e[1], "key": e[0]}, parent=evaluation_context, mutable=False)
            make_result = lambda results: {items[i][0]: results[i] for i in range(len(items))}
        else:
            # Iterable (non-str)
            items = list(collection)
            make_context = lambda e: EvaluationContext({"_": e}, parent=evaluation_context, mutable=False)
            make_result = lambda results: results

        # Step 2: Execute iteration (sequential or parallel)
        use_parallel = self._parallelism > 1 and len(items) > 1
        if use_parallel:
            intermediate_results = yield from self._execute_parallel(
                items, make_context, token_tracker, config
            )
        else:
            intermediate_results = yield from self._execute_sequential(
                items, make_context, token_tracker, config
            )

        # Step 3: Convert intermediate results to final result
        final_result = make_result(intermediate_results)

        if self._store_as:
            evaluation_context.add(self._store_as, final_result)

        return final_result

    def _execute_sequential(
            self,
            items: List[Any],
            make_context: Callable[[Any], EvaluationContext],
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig]
    ) -> Generator[WorkerNotification, None, List[Json]]:
        """Execute body sequentially for each item, returning list of results."""
        results = []
        for item in items:
            inner_context = make_context(item)
            result = yield from self._body_statement.yield_notifications_and_result(
                inner_context, token_tracker, config
            )
            results.append(result)
        return results

    def _execute_parallel(
            self,
            items: List[Any],
            make_context: Callable[[Any], EvaluationContext],
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig]
    ) -> Generator[WorkerNotification, None, List[Json]]:
        """Execute body in parallel for each item, returning list of results in original order."""
        notification_queue: queue.Queue = queue.Queue()
        results: Dict[int, Json] = {}
        has_exception = threading.Event()
        tasks_completed = 0
        total_tasks = len(items)

        def worker_task(index: int, item: Any) -> None:
            nonlocal tasks_completed
            try:
                inner_context = make_context(item)
                gen = self._body_statement.yield_notifications_and_result(
                    inner_context, token_tracker, config
                )

                # Consume the generator, pushing notifications to the queue
                result = None
                try:
                    while True:
                        notification = next(gen)
                        notification_queue.put(_IndexedNotification(index, notification))
                except StopIteration as e:
                    result = e.value

                notification_queue.put(_IndexedResult(index, result))

            except BaseException as e:
                notification_queue.put(_ExceptionSentinel(e, index))
                if not has_exception.is_set():
                    has_exception.set()
                    self._logger.debug(f"First exception in parallel for_each at index {index}: {e}")
                else:
                    self._logger.error(f"Additional exception in parallel for_each at index {index}: {e}")

            finally:
                tasks_completed += 1
                if tasks_completed >= total_tasks:
                    notification_queue.put(_CompletionSentinel())

        # Submit tasks to thread pool
        with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
            for idx, item in enumerate(items):
                if has_exception.is_set():
                    break
                executor.submit(worker_task, idx, item)

            # Yield notifications from the queue until all tasks complete
            while True:
                msg = notification_queue.get()

                if isinstance(msg, _CompletionSentinel):
                    break
                elif isinstance(msg, _IndexedNotification):
                    yield msg.notification
                elif isinstance(msg, _IndexedResult):
                    results[msg.index] = msg.result
                elif isinstance(msg, _ExceptionSentinel):
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise msg.exception

        # Reconstruct results in original order
        return [results[i] for i in range(total_tasks)]


class CustomTool(ExtendedExecutionTool):
    def __init__(self, context: WorkersContext, body: Statement, logger: Logger = _default_logger, **kwargs):
        super().__init__(**kwargs)
        self._default_evaluation_context = context.evaluation_context
        self._body = body
        self._logger = logger

    def default_evaluation_context(self) -> EvaluationContext:
        return self._default_evaluation_context

    # noinspection PyMethodOverriding
    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        input: Dict[str, Json],
        **kwargs: Any
    ) -> Generator[WorkerNotification, None, Any]:
        validated_input = self.args_schema(**input)
        evaluation_context = EvaluationContext(
            validated_input.model_dump(),
            parent=evaluation_context,
            logging_scope=self.name)
        return self._body.yield_notifications_and_result(evaluation_context, token_tracker, config)


def create_statement_from_model(model: StatementDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool], logger: Logger = _default_logger) -> Statement:
    if isinstance(model, EvalDefinition):
        return EvalStatement(model, logger)
    elif isinstance(model, CallDefinition):
        return CallStatement(model, context, local_tools, logger)
    elif isinstance(model, list):
        return FlowStatement(model, context, local_tools, logger)
    elif isinstance(model, IfDefinition):
        return IfStatement(model, context, local_tools, logger)
    elif isinstance(model, StarlarkDefinition):
        return StarlarkStatement(model, context, local_tools, logger)
    elif isinstance(model, ForEachDefinition):
        return ForEachStatement(model, context, local_tools, logger)
    else:
        raise ValueError(f"Invalid statement model type {type(model)}")


def create_dynamic_schema(name: str, params: List[CustomToolParamsDefinition]) -> Type[BaseModel]:
    # convert name to camel case
    cc_name = name.replace('_', ' ').title().replace(' ', '')
    model_name = f"{cc_name}DynamicSchema"
    fields = {}
    for param in params:
        field_type = parse_standard_type(param.type)
        coerce_num = True if param.type == 'str' else None
        if param.default is not None:
            fields[param.name] = (field_type, Field(description=param.description, default=param.default, coerce_numbers_to_str=coerce_num))
        else:
            fields[param.name] = (field_type, Field(description=param.description, coerce_numbers_to_str=coerce_num))
    return create_model(model_name, **fields)


def build_custom_tool(tool_def: CustomToolDefinition, context: WorkersContext) -> CustomTool:
    tools = context.get_tools(tool_def.name, tool_def.tools)
    local_tools = {tool.name: tool for tool in tools}
    logger = logging.getLogger(f"{__name__}.{tool_def.name}")
    body = create_statement_from_model(tool_def.do, context, local_tools, logger)

    return CustomTool(
        context=context,
        body=body,
        name=tool_def.name,
        description=tool_def.description,
        args_schema=create_dynamic_schema(tool_def.name, tool_def.input),
        return_direct=tool_def.return_direct or False,
        logger=logger
    )