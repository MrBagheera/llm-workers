import logging
import re
from copy import deepcopy, copy
from typing import Type, Any, Optional, Dict, TypeAlias, List, Iterator, AsyncIterator, Union

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field, create_model

from llm_workers.api import WorkersContext
from llm_workers.config import Json, CustomToolParamsDefinition, \
    CallDefinition, ResultDefinition, StatementDefinition, MatchDefinition, ToolDefinition
from llm_workers.utils import LazyFormatter, parse_standard_type

logger = logging.getLogger(__name__)


class TemplateHelper:
    """Helper tool to find templates in nested JSON structure and replace them during tool invocations."""

    direct_replacement_regexp: re.Pattern[str] = re.compile(r"\{([^{}]+)}")

    def __init__(self, replacement_map: dict[str, Any], target_params: dict[str, Json]):
        self._replacements = replacement_map
        self._target_params = target_params

    def _render(self, input_params: dict[str, Json], prefix: str, replacements: dict[Any, Any], target_params: dict[Any, Json] | list[Json]):
        for key, value in replacements.items():
            if isinstance(value, PromptTemplate):
                try:
                    target_params[key] = value.format(**input_params)
                except KeyError as ex:
                    raise ValueError(f"Missing reference for key {prefix}{key}: {ex}")
            elif isinstance(value, str):
                replacement = input_params.get(value)
                if replacement is None:
                    raise ValueError(f"Missing reference {{{value}}} for key {prefix}{key}")
                target_params[key] = input_params[value]
            else:
                self._render(input_params, prefix = f"{prefix}{key}.", replacements=value, target_params = target_params[key])

    def render(self, input_params: dict[str, Json]) -> dict[str, Json]:
        """Replaces template placeholders in target parameters with values from input parameters.

        Args:
            input_params: tool input parameters
        Returns:
            target parameters with rendered templates
        """
        if len(self._target_params) == 0:
            return self._target_params
        target_params = deepcopy(self._target_params)
        self._render(input_params, "", self._replacements, target_params)
        return target_params

    @classmethod
    def iter_items(cls, obj: Union[Dict[Any, Json], List[Json]]) -> Iterator[tuple[Any, Json]]:
        if isinstance(obj, dict):
            yield from obj.items()
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                yield index, item  # Use index as the key for lists

    @classmethod
    def build_replacement_map(cls, valid_template_vars: List[str], prefix: str, target_params: Iterator[tuple[Any, Json]]) -> dict:
        replacements = {}
        for key, value in target_params:
            if isinstance(value, str):
                match = re.fullmatch(cls.direct_replacement_regexp, value)
                if match:
                    reference = match.group(1)
                    if reference not in valid_template_vars:
                        raise ValueError(f"Unknown reference {{{reference}}} for key {prefix}{key}, available params: {valid_template_vars}")
                    replacements[key] = reference
                else:
                    prompt = PromptTemplate.from_template(value, template_format = "f-string")
                    if len(prompt.input_variables) > 0:
                        # validate all prompt inputs are in our params
                        for reference in prompt.input_variables:
                            if reference not in valid_template_vars:
                                raise ValueError(f"Unknown reference {{{reference}}} for key {prefix}{key}, available params: {valid_template_vars}")
                        replacements[key] = prompt
            elif isinstance(value, dict) or isinstance(value, list):
                sub_replacements = cls.build_replacement_map(valid_template_vars, f"{prefix}{key}.", cls.iter_items(value))
                if len(sub_replacements) > 0:
                    replacements[key] = sub_replacements
        return replacements

    @classmethod
    def from_valid_template_vars(cls, valid_template_vars: list[str], target_params: dict[str, Json]):
        replacement_map = cls.build_replacement_map(valid_template_vars, "", cls.iter_items(target_params))
        return TemplateHelper(replacement_map, target_params)

    @classmethod
    def from_param_definitions(cls, params: List[CustomToolParamsDefinition], target_params: dict[str, Json]):
        return cls.from_valid_template_vars([param.name for param in params], target_params)


Statement: TypeAlias = Runnable[Dict[str, Json], Json]


class ResultStatement(Runnable[Dict[str, Json], Json]):
    result_key: str = 'result'

    def __init__(self, valid_template_vars: List[str], model: ResultDefinition):
        params = { ResultStatement.result_key: model.result }
        self._template_helper = TemplateHelper.from_valid_template_vars(valid_template_vars, params)

    def invoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        result = self._template_helper.render(input)
        return result[self.result_key]

    async def ainvoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        result = self._template_helper.render(input)
        return result[self.result_key]


class CallStatement(Runnable[Dict[str, Json], Json]):

    def __init__(self, valid_template_vars: List[str], model: CallDefinition, context: WorkersContext):
        self._tool = context.get_tool(model.call)
        self._template_helper = TemplateHelper.from_valid_template_vars(valid_template_vars, model.params)
        if isinstance(model.catch, list):
            self._catch = model.catch
        elif isinstance(model.catch, str):
            self._catch = [model.catch]
        else:
            self._catch = None

    def invoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        target_params = self._template_helper.render(input)
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            return self._tool.invoke(input = target_params, config = config, **kwargs)
        except BaseException as e:
            raise self._convert_error(e)

    async def ainvoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        target_params = self._template_helper.render(input)
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            return await self._tool.ainvoke(input = target_params, config = config, **kwargs)
        except BaseException as e:
            raise self._convert_error(e)

    def stream(
        self,
        input: Dict[str, Json],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Json]:
        target_params = self._template_helper.render(input)
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            return self._tool.stream(input = target_params, config = config, **kwargs)
        except BaseException as e:
            raise self._convert_error(e)

    async def astream(
        self,
        input: Dict[str, Json],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Json]:
        target_params = self._template_helper.render(input)
        logger.debug("Calling tool %s with args:\n%r", self._tool.name, LazyFormatter(target_params))
        try:
            return self._tool.astream(input = target_params, config = config, **kwargs)
        except BaseException as e:
            raise self._convert_error(e)

    def _convert_error(self, e: BaseException) -> BaseException:
        if self._catch:
            exception_type = type(e).__name__
            for catch in self._catch:
                if catch == '*' or catch == 'all' or exception_type == catch:
                    return ToolException(str(e), e)
        return e



class FlowStatement(Runnable[Dict[str, Json], Json]):

    def __init__(self, valid_template_vars: List[str], model: list[StatementDefinition], context: WorkersContext):
        valid_template_vars = copy(valid_template_vars) # shallow copy is enough, we only append
        self._statements = []
        i = 0
        for statement_model in model:
            statement = create_statement_from_model(valid_template_vars, statement_model, context)
            self._statements.append(statement)
            valid_template_vars.append(f"output{i}") # Add reference to our output for next statements
            i += 1

    def invoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        input = copy(input) # shallow copy enough, we only append keys
        i = 0
        last = None
        for statement in self._statements:
            last = statement.invoke(input, config, **kwargs)
            input[f"output{i}"] = last
            i = i + 1
        return last

    async def ainvoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        input = copy(input) # shallow copy enough, we only append keys
        i = 0
        last = None
        for statement in self._statements:
            last = await statement.ainvoke(input, config, **kwargs)
            input[f"output{i}"] = last
            i = i + 1
        return input


class MatchStatement(Runnable[Dict[str, Json], Json]):
    match_key: str = 'match'

    def __init__(self, valid_template_vars: List[str], model: MatchDefinition, context: WorkersContext):
        self._template_helper = TemplateHelper.from_valid_template_vars(valid_template_vars, {MatchStatement.match_key: model.match})
        self._trim = model.trim
        self._clauses = []
        for matcher in model.matchers:
            if matcher.case:
                condition: str = matcher.case
                statement = create_statement_from_model(valid_template_vars, matcher.then, context)
                self._clauses.append((condition, statement))
            else:
                condition: re.Pattern[str] = re.compile(matcher.pattern)
                valid_clause_vars = copy(valid_template_vars) # shallow copy is enough, we only append
                for i in range(0, condition.groups):
                    valid_clause_vars.append(f"match{i}")
                statement = create_statement_from_model(valid_clause_vars, matcher.then, context)
                self._clauses.append((condition, statement))
        self._default = create_statement_from_model(valid_template_vars, model.default, context)

    def invoke(self, input: Dict[str, Json], config: Optional[RunnableConfig] = None, **kwargs: Any) -> Json:
        probe = self._template_helper.render(input)[self.match_key]
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
                    i = 0
                    input = copy(input) # shallow copy enough, we only append keys
                    for group in match.groups():
                        input[f"match{i}"] = group
                        i = i + 1
                    return statement.invoke(input, config, **kwargs)
            elif probe == condition:
                logger.debug("Probe [%s] matched condition [%s]", probe, condition)
                return statement.invoke(input, config, **kwargs)
        logger.debug("Probe [%s] did not match anything", probe)
        return self._default.invoke(input, config, **kwargs)


def create_statement_from_model(valid_template_vars: List[str], model: StatementDefinition, context: WorkersContext) -> Statement:
    if isinstance(model, ResultDefinition):
        return ResultStatement(valid_template_vars, model)
    elif isinstance(model, CallDefinition):
        return CallStatement(valid_template_vars, model, context)
    elif isinstance(model, list):
        return FlowStatement(valid_template_vars, model, context)
    elif isinstance(model, MatchDefinition):
        return MatchStatement(valid_template_vars, model, context)
    else:
        raise ValueError(f"Invalid statement model type {type(model)}")


def create_dynamic_schema(name: str, params: list[CustomToolParamsDefinition]) -> Type[BaseModel]:
    # convert name to camel case
    cc_name = name.replace('_', ' ').title().replace(' ', '')
    model_name = f"{cc_name}DynamicSchema"
    fields = {}
    for param in params:
        field_type = parse_standard_type(param.type)
        if param.default is not None:
            fields[param.name] = (field_type, Field(description=param.description, default=param.default, coerce_numbers_to_str=True))
        else:
            fields[param.name] = (field_type, Field(description=param.description, coerce_numbers_to_str=True))
    return create_model(model_name, **fields)


def build_custom_tool(definition: ToolDefinition, context: WorkersContext) -> StructuredTool:
    valid_template_vars = [param.name for param in definition.input]
    args_schema = create_dynamic_schema(definition.name, definition.input)
    body = create_statement_from_model(valid_template_vars, definition.body, context)

    def tool_logic(**input) -> Json:
        validated_input = args_schema(**input)
        return body.invoke(validated_input.model_dump(), **input)

    async def async_tool_logic(**input) -> Json:
        validated_input = args_schema(**input)
        return await body.ainvoke(validated_input.model_dump(), **input)

    return StructuredTool.from_function(
        func = tool_logic,
        coroutine = async_tool_logic,
        name = definition.name,
        description = definition.description,
        args_schema = create_dynamic_schema(definition.name, definition.input),
        return_direct = definition.return_direct or False,
    )