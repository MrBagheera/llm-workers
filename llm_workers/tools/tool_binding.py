from typing import Callable, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition, Json, build_dynamic_tool, TemplateHelper


class ToolBindingDefinition(CustomToolBaseDefinition):
    tool_ref: str
    tool_params: dict[str, Json]


def build_tool_binding(
    definition: ToolBindingDefinition,
    tools_lookup: Callable[[List[str]], List[BaseTool]]
) -> BaseTool:
    template_helper = TemplateHelper(definition, definition.tool_params)
    target = tools_lookup([definition.tool_ref])[0]

    def tool_logic(validated_input: BaseModel):
        target_params = template_helper.render(validated_input.model_dump())
        return target.invoke(input = target_params)

    async def async_tool_logic(validated_input: BaseModel):
        target_params = template_helper.render(validated_input.model_dump())
        return await target.ainvoke(input = target_params)

    return build_dynamic_tool(definition, tool_logic, async_tool_logic)
