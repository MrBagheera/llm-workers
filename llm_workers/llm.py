from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class BaseLLMConfig(BaseModel):
    provider: str
    model: str
    system_message: str = ""
    tool_refs: List[str]


def build_tool_calling_llm(llm_config: BaseLLMConfig, tools_lookup: callable) -> CompiledGraph:
    llm = init_chat_model(llm_config.model, model_provider=llm_config.provider, temperature=0)
    system_message = None
    if llm_config.system_message != "":
        system_message = SystemMessage(llm_config.system_message)
    return create_react_agent(llm, tools=tools_lookup(llm_config.tool_refs), state_modifier=system_message)
