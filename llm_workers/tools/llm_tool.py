from typing import Literal, Callable, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel

from llm_workers.llm import BaseLLMConfig, build_tool_calling_llm
from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition, build_dynamic_tool


class LLMToolDefinition(CustomToolBaseDefinition, BaseLLMConfig):
    type: Literal['LLM']
    prompt: str

def build_llm_tool(definition: LLMToolDefinition, models_lookup: Callable[[str], BaseChatModel], tools_lookup: Callable[[List[str]], List[BaseTool]]) -> BaseTool:
    prompt = PromptTemplate.from_template(definition.prompt)
    agent = build_tool_calling_llm(definition, models_lookup, tools_lookup)

    def tool_logic(validated_input: BaseModel):
        rendered_prompt = prompt.format(**validated_input.model_dump())
        return agent.invoke(input={"messages": [rendered_prompt]})["messages"][-1]

    async def async_tool_logic(validated_input: BaseModel):
        rendered_prompt = prompt.format(**validated_input.model_dump())
        result = await agent.ainvoke(input={"messages": [rendered_prompt]})
        return result["messages"][-1]

    return build_dynamic_tool(definition, tool_logic, async_tool_logic)
