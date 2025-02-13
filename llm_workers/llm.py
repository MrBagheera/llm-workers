from typing import List

from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from llm_workers.api import LLMWorkersContext


class BaseLLMConfig(BaseModel):
    model_ref: str = "default"
    system_message: str = None
    tool_refs: List[str] = ()


def build_tool_calling_llm(llm_config: BaseLLMConfig, context: LLMWorkersContext) -> CompiledGraph:
    llm = context.get_llm(llm_config.model_ref) # init_chat_model(llm_config.model, model_provider=llm_config.provider, temperature=0)
    system_message = None
    if llm_config.system_message is not None:
        system_message = SystemMessage(llm_config.system_message)
    tools = [context.get_tool(tool_name) for tool_name in llm_config.tool_refs]
    return create_react_agent(llm, tools=tools, state_modifier=system_message)
