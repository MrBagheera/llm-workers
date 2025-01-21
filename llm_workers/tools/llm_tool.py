from typing import Literal, List

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from llm_workers.tools.custom_tools_base import CustomToolBaseConfig, build_dynamic_tool


class LLMToolConfig(CustomToolBaseConfig):
    tool_type: Literal['LLM']
    tool_refs: List[str] = ()
    instructions: str


def build_llm_tool(tool_config: LLMToolConfig, tool_lookup: callable) -> BaseTool:
    def llm_tool_logic(validated_input: BaseModel):
        raise ToolException("LLMTool is not implemented")

    return build_dynamic_tool(tool_config, llm_tool_logic)
