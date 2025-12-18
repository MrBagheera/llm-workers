import logging
import re
from copy import deepcopy, copy
from typing import Type, Any, Optional, Dict, TypeAlias, List, Iterator, Union, Iterable

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, create_model, PrivateAttr

from llm_workers.api import WorkersContext, WorkerNotification, ExtendedRunnable, ExtendedExecutionTool
from llm_workers.config import Json, CustomToolParamsDefinition, \
    CallDefinition, EvalDefinition, StatementDefinition, MatchDefinition, CustomToolDefinition
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import LazyFormatter, parse_standard_type
from llm_workers.worker_utils import call_tool

logger = logging.getLogger(__name__)


Statement: TypeAlias = ExtendedRunnable[Dict[str, Json], Json]


# noinspection PyTypeHints
class EvalStatement(ExtendedRunnable[Dict[str, Json], Json]):
    def __init__(self, model: EvalDefinition):
        self._eval_expr = model.eval

    def _stream(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Iterable[Any]:
        result = self._eval_expr.evaluate(evaluation_context)
        yield result


# noinspection PyTypeHints
class CallStatement(ExtendedRunnable[Dict[str, Json], Json]):

    def __init__(self, model: CallDefinition, context: WorkersContext):
        self._tool = context.get_tool(model.call)
        self._params_expr = model.params
        if isinstance(model.catch, list):
            self._catch = model.catch
        elif isinstance(model.catch, str):
            self._catch = [model.catch]
        else:
            self._catch = None

    def _stream(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Iterable[Any]:
        # Evaluate params expression
        target_params = self._params_expr.evaluate(evaluation_context) if self._params_expr else {}
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            result = None
            for chunk in call_tool(self._tool, input=target_params, evaluation_context=evaluation_context, token_tracker=token_tracker, config=config, kwargs={}):
                if isinstance(chunk, WorkerNotification):
                    yield chunk
                else:
                    result = chunk
            logger.debug("Calling tool %s resulted:\n%r", self._tool.name, LazyFormatter(result, trim=False))
            yield result
        except BaseException as e:
            raise self._convert_error(e)

    def _convert_error(self, e: BaseException) -> BaseException:
        if self._catch:
            exception_type = type(e).__name__
            for catch in self._catch:
                if catch == '*' or catch == 'all' or exception_type == catch:
                    return ToolException(str(e), e)
        return e


class FlowStatement(ExtendedRunnable[Dict[str, Json], Json]):

    def __init__(self, model: list[StatementDefinition], context: WorkersContext):
        self._statements = []
        i = 0
        for statement_model in model:
            statement = create_statement_from_model(statement_model, context)
            self._statements.append(statement)
            i += 1

    def _stream(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Iterable[Any]:
        i = 0
        last = None
        for statement in self._statements:
            for chunk in statement._stream(evaluation_context, token_tracker, config):
                if isinstance(chunk, WorkerNotification):
                    yield chunk
                else:
                    last = chunk
                    break
            evaluation_context.add(f"output{i}", last)
            evaluation_context.add("_", last)
            i = i + 1
        yield last


# noinspection PyTypeHints
class MatchStatement(ExtendedRunnable[Dict[str, Json], Json]):
    match_key: str = 'match'

    def __init__(self, model: MatchDefinition, context: WorkersContext):
        self._match_expr = model.match
        self._trim = model.trim
        self._clauses = []
        for matcher in model.matchers:
            if matcher.case:
                condition: str = matcher.case
                statement = create_statement_from_model(matcher.then, context)
                self._clauses.append((condition, statement))
            else:
                condition: re.Pattern[str] = re.compile(matcher.pattern)
                statement = create_statement_from_model(matcher.then, context)
                self._clauses.append((condition, statement))
        self._default = create_statement_from_model(model.default, context)

    def _stream(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any   # ignored
    ) -> Iterable[Any]:
        probe = self._match_expr.evaluate(evaluation_context)
        probe_str = None
        if self._trim:
            probe = str(probe).strip()
            probe_str = probe
        for condition, statement in self._clauses:
            if isinstance(condition, re.Pattern):
                if not probe_str:
                    probe_str = str(probe)
                match = condition.fullmatch(probe_str)
                if match:
                    logger.debug("Probe [%s] matched regexp [%s]", probe_str, condition)
                    evaluation_context.add('match', match.groups())
                    yield from statement._stream(
                        EvaluationContext({'match': match.groups()}, parent=evaluation_context),
                        token_tracker,
                        config)
                    return
            elif probe == condition:
                logger.debug("Probe [%s] matched condition [%s]", probe, condition)
                yield from statement._stream(
                    EvaluationContext({}, parent=evaluation_context),
                    token_tracker,
                    config)
                return
        logger.debug("Probe [%s] did not match anything", probe)
        yield from self._default._stream(
            EvaluationContext({}, parent=evaluation_context),
            token_tracker,
            config)


class CustomTool(ExtendedExecutionTool):
    _context: WorkersContext = PrivateAttr()
    _body: ExtendedRunnable[dict[str, Any], Any] = PrivateAttr()

    def __init__(self, context: WorkersContext, body: ExtendedRunnable[dict[str, Any], Any], **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._body = body

    def default_evaluation_context(self) -> EvaluationContext:
        return self._context.evaluation_context

    def _stream(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: Optional[CompositeTokenUsageTracker],
        config: Optional[RunnableConfig],
        **kwargs: Any
    ) -> Iterable[Any]:
        validated_input = self.args_schema(**kwargs)
        # starting new evaluation context
        evaluation_context = EvaluationContext(validated_input.model_dump(), parent=evaluation_context)
        yield from self._body._stream(evaluation_context, token_tracker, config, **(validated_input.model_extra or {}))


def create_statement_from_model(model: StatementDefinition, context: WorkersContext) -> Statement:
    if isinstance(model, EvalDefinition):
        return EvalStatement(model)
    elif isinstance(model, CallDefinition):
        return CallStatement(model, context)
    elif isinstance(model, list):
        return FlowStatement(model, context)
    elif isinstance(model, MatchDefinition):
        return MatchStatement(model, context)
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
    body = create_statement_from_model(tool_def.body, context)

    return CustomTool(
        context=context,
        body=body,
        name=tool_def.name,
        description=tool_def.description,
        args_schema=create_dynamic_schema(tool_def.name, tool_def.input),
        return_direct=tool_def.return_direct or False
    )