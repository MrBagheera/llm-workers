from typing import Literal, List

from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field

from llm_workers.tools.custom_tools_base import CustomToolBaseConfig, build_dynamic_tool


class MatcherModel(BaseModel):
    case: str
    result: str = Field(coerce_numbers_to_str=True)

class DebugToolConfig(CustomToolBaseConfig):
    tool_type: Literal['debug']
    match_value: str
    matchers: List[MatcherModel]
    default_result: str


def build_debug_tool(tool_config: DebugToolConfig) -> BaseTool:
    prompt = PromptTemplate.from_template(tool_config.match_value)

    def debug_tool_logic(validated_input: BaseModel):
        match_value = prompt.format(**validated_input.model_dump())
        for matcher in tool_config.matchers:
            if matcher.case == match_value:
                return {"result": matcher.result}
        return {"result": tool_config.default_result}

    return build_dynamic_tool(tool_config, debug_tool_logic)

