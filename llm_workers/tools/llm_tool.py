from typing import Literal, Dict, Any

from langchain_core.tools import BaseTool, StructuredTool

from llm_workers.api import LLMWorkersContext
from llm_workers.llm import BaseLLMConfig, build_tool_calling_llm


def build_llm_tool(context: LLMWorkersContext, raw_config: Dict[str, Any]) -> BaseTool:
    config = BaseLLMConfig(**raw_config)
    agent = build_tool_calling_llm(config, context)

    def tool_logic(prompt: str):
        """
        Calls LLM with given prompt, returns LLM output.

        Args:
            prompt: text prompt
        """
        return agent.invoke(input={"messages": [prompt]})["messages"][-1]

    async def async_tool_logic(prompt: str):
        result = await agent.ainvoke(input={"messages": [prompt]})
        return result["messages"][-1]

    return StructuredTool.from_function(
        func = tool_logic,
        coroutine=async_tool_logic,
        name='llm',
        parse_docstring=True,
        error_on_invalid_docstring=True
    )
