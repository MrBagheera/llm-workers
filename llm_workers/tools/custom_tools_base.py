from typing import Type, Dict, Any

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field, create_model


class ToolParamConfig(BaseModel):
    name: str
    description: str
    type: str


class CustomToolBaseConfig(BaseModel):
    name: str
    description: str
    params: list[ToolParamConfig]


def create_dynamic_schema(params: list[ToolParamConfig]) -> Type[BaseModel]:
    fields = {}
    for param in params:
        field_type = eval(param.type)  # Convert string type to Python type
        fields[param.name] = (field_type, Field(field_type, description=param.description, coerce_numbers_to_str=True))
    return create_model("DynamicSchema", **fields)


def build_dynamic_tool(tool_config: CustomToolBaseConfig, tool_logic: callable) -> StructuredTool:
    schema = create_dynamic_schema(tool_config.params)

    def wrapped_tool_logic(**kwargs: Any) -> Any:
        # Validate inputs using the generated schema
        validated_input = schema(**kwargs)
        return tool_logic(validated_input)

    return StructuredTool.from_function(
        func=wrapped_tool_logic,
        name=tool_config.name,
        description=tool_config.description,
        args_schema=schema
    )
