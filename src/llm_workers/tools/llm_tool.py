from typing import Dict, Any, List, Union
import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from llm_workers.api import WorkersContext
from llm_workers.config import BaseLLMConfig, ToolLLMConfig
from llm_workers.worker import Worker


def extract_json_blocks(text: str, extract_json: Union[bool, str]) -> str:
    """
    Extract JSON blocks from text based on the extract_json parameter.
    
    Args:
        text: The input text to extract JSON from
        extract_json: Filtering option - True/"first", "last", "all", or "none"
        
    Returns:
        Extracted JSON as string or original text if no JSON found
    """
    if extract_json is None or extract_json == "none" or extract_json is False:
        return text
    
    # Find all JSON blocks (objects and arrays)
    json_pattern = r'```json\s*\n(.*?)\n```|```\s*\n(\{.*?\}|\[.*?\])\n```|(\{.*?\}|\[.*?\])'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        # Fallback: try to find JSON-like structures without code blocks
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
        matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        return text  # Fallback to full message if no JSON found
    
    # Extract non-empty matches
    json_blocks = []
    for match in matches:
        if isinstance(match, tuple):
            # Find the non-empty group from the tuple
            for group in match:
                if group.strip():
                    json_blocks.append(group.strip())
                    break
        else:
            json_blocks.append(match.strip())
    
    if not json_blocks:
        return text
    
    # Apply filtering
    if extract_json is True or extract_json == "first":
        return json_blocks[0]
    elif extract_json == "last":
        return json_blocks[-1]
    elif extract_json == "all":
        return json.dumps(json_blocks)
    
    return text


def build_llm_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> BaseTool:
    config = ToolLLMConfig(**tool_config)
    agent = Worker(config, context)

    def extract_result(result: List[BaseMessage]) -> str:
        if len(result) == 0:
            return ""
        if len(result) == 1:
            text = str(result[0].text())
        elif len(result) > 1:
            # return only AI message(s)
            text = "\n".join([message.text() for message in result if isinstance(message, AIMessage)])
        else:
            text = ""
        
        # Apply JSON filtering if configured
        return extract_json_blocks(text, config.extract_json)

    def tool_logic(prompt: str, system_message: str = None) -> str:
        """
        Calls LLM with given prompt, returns LLM output.

        Args:
            prompt: text prompt
            system_message: optional system message to prepend to the conversation
        """
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(prompt))
        result = agent.invoke(input=messages)
        return extract_result(result)

    async def async_tool_logic(prompt: str, system_message: str = None) -> str:
        # pass empty callbacks to prevent LLM token streaming
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(prompt))
        result = await agent.ainvoke(input=messages)
        return extract_result(result)

    return StructuredTool.from_function(
        func = tool_logic,
        coroutine=async_tool_logic,
        name='llm',
        parse_docstring=True,
        error_on_invalid_docstring=True
    )
