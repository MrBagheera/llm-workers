from typing import Literal, List

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition, build_dynamic_tool, Json


class MatcherConfig(BaseModel):
    case: str
    result: Json

class StubToolDefinition(CustomToolBaseDefinition):
    type: Literal['stub']
    match_value: str
    matchers: List[MatcherConfig]
    default_result: Json

def build_stub_tool(definition: StubToolDefinition) -> BaseTool:
    prompt = PromptTemplate.from_template(definition.match_value)

    def tool_logic(validated_input: BaseModel):
        match_value = prompt.format(**validated_input.model_dump())
        for matcher in definition.matchers:
            if matcher.case == match_value:
                return matcher.result
        return definition.default_result

    async def async_tool_logic(validated_input: BaseModel):
        return tool_logic(validated_input)

    return build_dynamic_tool(definition, tool_logic, async_tool_logic)

