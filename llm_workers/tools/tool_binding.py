from copy import deepcopy
from typing import Callable, List

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition, Json, build_dynamic_tool


class ToolBindingDefinition(CustomToolBaseDefinition):
    tool_ref: str
    tool_params: dict[str, Json]


def _find_inner_tool_param_templates(definition: ToolBindingDefinition) -> dict[str, PromptTemplate]:
    templates = {}
    valid_template_vars = [param.name for param in definition.params]
    for tool_param, value in definition.tool_params.items():
        if isinstance(value, str):
            prompt = PromptTemplate.from_template(value)
            if len(prompt.input_variables) > 0:
                # validate all prompt inputs are in our params
                for variable in prompt.input_variables:
                    if variable not in valid_template_vars:
                        raise ValueError(f"Unknown prompt variable {variable}, available params: {valid_template_vars}")
                templates[tool_param] = prompt
    return templates

def build_tool_binding(definition: ToolBindingDefinition, tools_lookup: Callable[[List[str]], List[BaseTool]]) -> BaseTool:
    target_param_templates = _find_inner_tool_param_templates(definition)
    target = tools_lookup([definition.tool_ref])[0]

    def build_target_params(validated_input: BaseModel):
        target_params = deepcopy(definition.tool_params)
        raw_input = validated_input.model_dump()
        for key, value in definition.tool_params.items():
            template = target_param_templates.get(key, None)
            if template is not None:
                target_params[key] = template.format(**raw_input)
        return target_params

    def tool_logic(validated_input: BaseModel):
        target_params = build_target_params(validated_input)
        return target.invoke(input = target_params)

    async def async_tool_logic(validated_input: BaseModel):
        target_params = build_target_params(validated_input)
        return target.invoke(input = target_params)

    return build_dynamic_tool(definition, tool_logic, async_tool_logic)
