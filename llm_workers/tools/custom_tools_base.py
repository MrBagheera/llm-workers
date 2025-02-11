from copy import deepcopy
from typing import Type, Any, TypeAliasType, Annotated, Union, Callable, Awaitable

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model, ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo


def json_custom_error_validator(
    value: Any,
    handler: ValidatorFunctionWrapHandler,
    _info: ValidationInfo
) -> Any:
    """Simplify the error message to avoid a gross error stemming
    from exhaustive checking of all union options.
    """
    try:
        return handler(value)
    except ValidationError:
        raise PydanticCustomError(
            'invalid_json',
            'Input is not valid json',
        )


Json = TypeAliasType(
    'Json',
    Annotated[
        Union[dict[str, 'Json'], list['Json'], str, int, float, bool, None],
        WrapValidator(json_custom_error_validator),
    ],
)




class CustomToolParamsDefinition(BaseModel):
    name: str
    description: str
    type: str
    default: Json | None = None


class CustomToolBaseDefinition(BaseModel):
    name: str
    description: str
    params: list[CustomToolParamsDefinition]
    return_direct: bool = False


def create_dynamic_schema(name: str, params: list[CustomToolParamsDefinition]) -> Type[BaseModel]:
    # convert name to camel case
    cc_name = name.replace('_', ' ').title().replace(' ', '')
    model_name = f"{cc_name}DynamicSchema"
    fields = {}
    for param in params:
        field_type = eval(param.type)  # Convert string type to Python type
        if param.default is not None:
            fields[param.name] = (field_type, Field(description=param.description, default=param.default, coerce_numbers_to_str=True))
        else:
            fields[param.name] = (field_type, Field(description=param.description, coerce_numbers_to_str=True))
    return create_model(model_name, **fields)


def build_dynamic_tool(
    definition: CustomToolBaseDefinition,
    tool_logic: Callable,
    async_tool_logic: Callable[..., Awaitable[Any]] | None
) -> StructuredTool:
    schema = create_dynamic_schema(definition.name, definition.params)

    def wrapped_tool_logic(**kwargs: Any) -> Any:
        validated_input = schema(**kwargs)
        return tool_logic(validated_input)

    async def async_wrapped_tool_logic(**kwargs: Any) -> Any:
        validated_input = schema(**kwargs)
        return await async_tool_logic(validated_input)

    return StructuredTool.from_function(
        func=wrapped_tool_logic,
        coroutine=None if async_tool_logic is None else async_wrapped_tool_logic,
        name=definition.name,
        description=definition.description,
        args_schema=schema,
        return_direct=definition.return_direct
    )


class TemplateHelper:
    """Helper tool to find templates in nested JSON structure and replace them during tool invocations."""

    def __init__(self, definition: CustomToolBaseDefinition, target_params: dict[str, Json]):
        """Constructor.
        TODO support templates in nested parameters

        Args:
            definition (CustomToolBaseDefinition): tool definition to take input parameters from
            target_params (dict[str, Json]): target set of parameters to search for template patterns
        """
        self._templates = {}
        valid_template_vars = [param.name for param in definition.params]
        for tool_param, value in target_params.items():
            if isinstance(value, str):
                prompt = PromptTemplate.from_template(value)
                if len(prompt.input_variables) > 0:
                    # validate all prompt inputs are in our params
                    for variable in prompt.input_variables:
                        if variable not in valid_template_vars:
                            raise ValueError(f"Unknown prompt variable {variable}, available params: {valid_template_vars}")
                    self._templates[tool_param] = prompt
        self._target_params = target_params

    def render(self, input_params: dict[str, Json]) -> dict[str, Json]:
        """Replaces template placeholders in target parameters with values from input parameters.

        Args:
            input_params (dict[str, Json]): tool input parameters
        Returns:
            dict[str, Json]: target parameters with rendered templates
        """
        if len(self._target_params) == 0:
            return self._target_params
        target_params = deepcopy(self._target_params)
        for key, template in self._templates.items():
            target_params[key] = template.format(**input_params)
        return target_params
