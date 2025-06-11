from typing import Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool, StructuredTool

from llm_workers.api import WorkersContext
from llm_workers.config import BaseLLMConfig
from llm_workers.worker import Worker


def build_llm_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> BaseTool:
    config = BaseLLMConfig(**tool_config)
    agent = Worker(config, context)

    def extract_result(result: List[BaseMessage]) -> str:
        if len(result) == 0:
            return ""
        if len(result) == 1:
            return str(result[0].text())
        if len(result) > 1:
            # return only AI message(s)
            return "\n".join([message.text() for message in result if isinstance(message, AIMessage)])

    def tool_logic(prompt: str) -> str:
        """
        Calls LLM with given prompt, returns LLM output.

        Args:
            prompt: text prompt
        """
        result = agent.invoke(input=[HumanMessage(prompt)])
        return extract_result(result)

    async def async_tool_logic(prompt: str) -> str:
        # pass empty callbacks to prevent LLM token streaming
        result = await agent.ainvoke(input=[HumanMessage(prompt)])
        return extract_result(result)

    return StructuredTool.from_function(
        func = tool_logic,
        coroutine=async_tool_logic,
        name='llm',
        parse_docstring=True,
        error_on_invalid_docstring=True
    )
