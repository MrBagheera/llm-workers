import logging
from typing import Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, StructuredTool

from llm_workers.api import WorkersContext
from llm_workers.llm import BaseLLMConfig, build_tool_calling_llm

logger = logging.getLogger(__name__)

def build_llm_tool(context: WorkersContext, raw_config: Dict[str, Any]) -> BaseTool:
    config = BaseLLMConfig(**raw_config)
    agent = build_tool_calling_llm(config, context)

    def extract_result(result: Dict[str, Any]) -> str:
        result = result["messages"][-1]
        if isinstance(result, BaseMessage):
            return result.content
        logger.warning(f"Unexpected result type: {type(result)}: {result}")
        return str(result)

    def tool_logic(prompt: str):
        """
        Calls LLM with given prompt, returns LLM output.

        Args:
            prompt: text prompt
        """
        result = agent.invoke(input={"messages": [prompt]})
        return extract_result(result)

    async def async_tool_logic(prompt: str):
        result = await agent.ainvoke(input={"messages": [prompt]})
        return extract_result(result)

    return StructuredTool.from_function(
        func = tool_logic,
        coroutine=async_tool_logic,
        name='llm',
        parse_docstring=True,
        error_on_invalid_docstring=True
    )
