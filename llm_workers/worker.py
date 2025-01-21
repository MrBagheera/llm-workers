from langchain.chat_models import init_chat_model
from langchain.globals import set_debug
from langchain.globals import set_verbose
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from llm_workers.config import load_config
from llm_workers.tools.tool_registry import ToolRegistry


def build_worker(config_filename: str):
    config = load_config(config_filename)
    set_verbose(config.main.debug)
    set_debug(config.main.debug)
    tool_registry = ToolRegistry()
    llm = init_chat_model(config.main.model, model_provider=config.main.provider, temperature=0)
    return create_react_agent(llm, tools=tool_registry.resolve_tool_refs(config.main.tool_refs), state_modifier=SystemMessage(config.main.instructions))
