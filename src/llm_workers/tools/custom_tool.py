import logging
import re
from copy import deepcopy, copy
from typing import Type, Any, Optional, Dict, TypeAlias, List, Iterator, Union, Iterable, Generator

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, create_model, PrivateAttr

from llm_workers.api import WorkersContext, WorkerNotification, ExtendedRunnable, ExtendedExecutionTool
from llm_workers.config import Json, CustomToolParamsDefinition, \
    CallDefinition, EvalDefinition, StatementDefinition, IfDefinition, CustomToolDefinition
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
            token_tracker: Optional[CompositeTokenUsageTracker],
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

    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: Optional[CompositeTokenUsageTracker],
        config: Optional[RunnableConfig],
        **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        # Evaluate params expression
        target_params = self._params_expr.evaluate(evaluation_context) if self._params_expr else {}
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            result = yield from call_tool(self._tool, target_params, evaluation_context, token_tracker, config, kwargs)
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
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Generator[WorkerNotification, None, Json]:
        result = None
        for statement in self._statements:
            inner_context = EvaluationContext({"_": result}, parent=evaluation_context, mutable=False)
            result = yield from statement.yield_notifications_and_result(inner_context, token_tracker, config)
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
            token_tracker: Optional[CompositeTokenUsageTracker],
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


class CustomTool(ExtendedExecutionTool):
    def __init__(self, context: WorkersContext, body: Statement, **kwargs):
        super().__init__(**kwargs)
        self._default_evaluation_context = context.evaluation_context
        self._body = body

    def default_evaluation_context(self) -> EvaluationContext:
        return self._default_evaluation_context

    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: Optional[CompositeTokenUsageTracker],
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


def build_custom_tool(tool_def: CustomToolDefinition, context: WorkersContext) -> StructuredTool:
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