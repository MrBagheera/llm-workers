import logging
from collections.abc import Iterable
from typing import Type, Any, Optional, Dict, TypeAlias, List, Generator

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, create_model

from llm_workers.api import WorkersContext, WorkerNotification, ExtendedRunnable, ExtendedExecutionTool
from llm_workers.config import Json, CustomToolParamsDefinition, \
    CallDefinition, EvalDefinition, StatementDefinition, IfDefinition, StarlarkDefinition, ForEachDefinition, CustomToolDefinition
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import LazyFormatter, parse_standard_type
from llm_workers.worker_utils import call_tool

logger = logging.getLogger(__name__)


Statement: TypeAlias = ExtendedRunnable[Json]


class EvalStatement(ExtendedRunnable[Json]):
    def __init__(self, model: EvalDefinition):
        self._eval_expr = model.eval
        self._store_as = model.store_as

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        result = self._eval_expr.evaluate(evaluation_context)
        if False:  # To make this function return generator, yield statement must exists in it's body
            yield WorkerNotification()
        if self._store_as:
            evaluation_context.add(self._store_as, result)
        return result # cannot use return here due to generator


# noinspection PyTypeHints
class CallStatement(ExtendedRunnable[Json]):

    def __init__(self, model: CallDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool]):
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

    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        # Evaluate params expression
        target_params = self._params_expr.evaluate(evaluation_context) if self._params_expr else {}
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            result = yield from call_tool(self._tool, target_params, evaluation_context, token_tracker, config, kwargs, ui_hint_override=self._ui_hint)
            logger.debug("Calling tool %s resulted:\n%r", self._tool.name, LazyFormatter(result, trim=False))
            if self._store_as:
                evaluation_context.add(self._store_as, result)
            return result
        except BaseException as e:
            raise self._convert_error(e)

    def _convert_error(self, e: BaseException) -> BaseException:
        if self._catch:
            exception_type = type(e).__name__
            for catch in self._catch:
                if catch == '*' or catch == 'all' or exception_type == catch:
                    return ToolException(str(e), e)
        return e


class FlowStatement(ExtendedRunnable[Json]):

    def __init__(self, model: list[StatementDefinition], context: WorkersContext, local_tools: Dict[str, BaseTool]):
        self._statements: List[Statement] = []
        for statement_model in model:
            statement = create_statement_from_model(statement_model, context, local_tools)
            self._statements.append(statement)

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
            logger.debug("Flow statement at %s yielded:\n%r", i, LazyFormatter(result, trim=False))
            i += 1
        return result


# noinspection PyTypeHints
class IfStatement(ExtendedRunnable[Json]):
    """
    Executes conditional logic based on boolean expression evaluation.

    If the condition evaluates to a truthy value, executes the 'then' branch.
    Otherwise, executes the 'else' branch (if provided) or returns None.
    """

    def __init__(self, model: IfDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool]):
        self._condition_expr = model.if_
        self._then_statement = create_statement_from_model(model.then, context, local_tools)
        self._else_statement = None
        if model.else_ is not None:
            self._else_statement = create_statement_from_model(model.else_, context, local_tools)
        self._store_as = model.store_as

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
            logger.debug("If condition [%s] evaluated to truthy, executing 'then' branch", condition_result)
            result = yield from self._then_statement.yield_notifications_and_result(
                evaluation_context, token_tracker, config
            )
        elif self._else_statement is not None:
            logger.debug("If condition [%s] evaluated to falsy, executing 'else' branch", condition_result)
            result = yield from self._else_statement.yield_notifications_and_result(
                evaluation_context, token_tracker, config
            )
        else:
            logger.debug("If condition [%s] evaluated to falsy, no 'else' branch, returning None", condition_result)
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

    def __init__(self, model: StarlarkDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool]):
        self._script = model.starlark
        self._store_as = model.store_as
        # combine local and shared tools (local take precedence)
        self._tools = context.shared_tools | local_tools

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
        result = self._executor.run(global_vars, global_funcs)

        # Store result if requested
        if self._store_as:
            evaluation_context.add(self._store_as, result)

        # Dummy yield to make this a generator
        if False:
            yield WorkerNotification()

        return result


class ForEachStatement(ExtendedRunnable[Json]):
    """
    Iterates over a collection and executes the body for each element.

    - Dict: Maps over values (preserving keys), returns dict
    - Iterable (except str): Maps over elements, returns list of results
    - Other (including str): Treats as single item, returns single result

    Available in body: `_` = current element, `key` = key (for dicts)
    """

    def __init__(self, model: ForEachDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool]):
        self._collection_expr = model.for_each
        self._body_statement = create_statement_from_model(model.do, context, local_tools)
        self._store_as = model.store_as

    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Generator[WorkerNotification, None, Json]:
        collection = self._collection_expr.evaluate(evaluation_context)

        if isinstance(collection, dict):
            # Dict: map over values, preserve keys
            results = {}
            for key, value in collection.items():
                inner_context = EvaluationContext({"_": value, "key": key}, parent=evaluation_context, mutable=False)
                result = yield from self._body_statement.yield_notifications_and_result(
                    inner_context, token_tracker, config
                )
                results[key] = result
            final_result = results

        elif isinstance(collection, Iterable) and not isinstance(collection, str):
            # Any iterable (except str): map over elements, return list
            results = []
            for item in collection:
                inner_context = EvaluationContext({"_": item}, parent=evaluation_context, mutable=False)
                result = yield from self._body_statement.yield_notifications_and_result(
                    inner_context, token_tracker, config
                )
                results.append(result)
            final_result = results

        else:
            inner_context = EvaluationContext({"_": collection}, parent=evaluation_context, mutable=False)
            final_result = yield from self._body_statement.yield_notifications_and_result(
                inner_context, token_tracker, config
            )

        if self._store_as:
            evaluation_context.add(self._store_as, final_result)

        return final_result


class CustomTool(ExtendedExecutionTool):
    def __init__(self, context: WorkersContext, body: Statement, **kwargs):
        super().__init__(**kwargs)
        self._default_evaluation_context = context.evaluation_context
        self._body = body

    def default_evaluation_context(self) -> EvaluationContext:
        return self._default_evaluation_context

    # noinspection PyMethodOverriding
    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        input: dict[str, Json],
        **kwargs: Any
    ) -> Generator[WorkerNotification, None, Any]:
        validated_input = self.args_schema(**input)
        # starting new evaluation context
        evaluation_context = EvaluationContext(validated_input.model_dump(), parent=evaluation_context)
        return self._body.yield_notifications_and_result(evaluation_context, token_tracker, config)


def create_statement_from_model(model: StatementDefinition, context: WorkersContext, local_tools: Dict[str, BaseTool]) -> Statement:
    if isinstance(model, EvalDefinition):
        return EvalStatement(model)
    elif isinstance(model, CallDefinition):
        return CallStatement(model, context, local_tools)
    elif isinstance(model, list):
        return FlowStatement(model, context, local_tools)
    elif isinstance(model, IfDefinition):
        return IfStatement(model, context, local_tools)
    elif isinstance(model, StarlarkDefinition):
        return StarlarkStatement(model, context, local_tools)
    elif isinstance(model, ForEachDefinition):
        return ForEachStatement(model, context, local_tools)
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
    body = create_statement_from_model(tool_def.do, context, local_tools)

    return CustomTool(
        context=context,
        body=body,
        name=tool_def.name,
        description=tool_def.description,
        args_schema=create_dynamic_schema(tool_def.name, tool_def.input),
        return_direct=tool_def.return_direct or False
    )