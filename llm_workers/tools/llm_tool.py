from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from llm_workers.llm import BaseLLMConfig, build_tool_calling_llm
from llm_workers.tools.custom_tools_base import CustomToolBaseConfig, build_dynamic_tool


class LLMToolConfig(CustomToolBaseConfig, BaseLLMConfig):
    tool_type: Literal['LLM']
    prompt: str


def build_llm_tool(tool_config: LLMToolConfig, tools_lookup: callable) -> BaseTool:
    prompt = PromptTemplate.from_template(tool_config.prompt)
    agent = build_tool_calling_llm(tool_config, tools_lookup)

    def llm_tool_logic(validated_input: BaseModel):
        rendered_prompt = prompt.format(**validated_input.model_dump())
        return agent.invoke(input={"messages": [rendered_prompt]})["messages"][-1]

    return build_dynamic_tool(tool_config, llm_tool_logic)
