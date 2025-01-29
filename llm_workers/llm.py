from typing import List, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class BaseLLMConfig(BaseModel):
    model_ref: str = "default"
    system_message: str = None
    tool_refs: List[str] = ()


def build_tool_calling_llm(llm_config: BaseLLMConfig, models_lookup: Callable[[str], BaseChatModel], tools_lookup: Callable[[List[str]], List[BaseTool]]) -> CompiledGraph:
    llm = models_lookup(llm_config.model_ref) # init_chat_model(llm_config.model, model_provider=llm_config.provider, temperature=0)
    system_message = None
    if llm_config.system_message is not None:
        system_message = SystemMessage(llm_config.system_message)
    return create_react_agent(llm, tools=tools_lookup(llm_config.tool_refs), state_modifier=system_message)
