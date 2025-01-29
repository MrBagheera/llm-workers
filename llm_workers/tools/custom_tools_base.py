from typing import Type, Any, TypeAliasType, Annotated, Union, Optional, Callable, Awaitable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model, ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo


class CustomToolParamsDefinition(BaseModel):
    name: str
    description: str
    type: str


class CustomToolBaseDefinition(BaseModel):
    name: str
    description: str
    params: list[CustomToolParamsDefinition]


def json_custom_error_validator(
        value: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
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


def create_dynamic_schema(name: str, params: list[CustomToolParamsDefinition]) -> Type[BaseModel]:
    # convert name to camel case
    cc_name = name.replace('_', ' ').title().replace(' ', '')
    model_name = f"{cc_name}DynamicSchema"
    fields = {}
    for param in params:
        field_type = eval(param.type)  # Convert string type to Python type
        fields[param.name] = (field_type, Field(field_type, description=param.description, coerce_numbers_to_str=True))
    return create_model(model_name, **fields)


def build_dynamic_tool(
    definition: CustomToolBaseDefinition,
    tool_logic: Callable,
    async_tool_logic: Callable[..., Awaitable[Any]]
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
        coroutine=async_wrapped_tool_logic,
        name=definition.name,
        description=definition.description,
        args_schema=schema
    )
