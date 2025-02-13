import importlib
import logging
from typing import Dict, Optional, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

import llm_workers.tools.llm_tool
from llm_workers.api import LLMWorkersContext
from llm_workers.config import WorkerConfig
from llm_workers.tools.custom_tool import build_custom_tool

logger = logging.getLogger(__name__)


class StandardContext(LLMWorkersContext):

    # Factory methods for predefined tools, used for resolving tool references
    # with dependencies only if the tool is actually used
    _predefined_tools_factories = {
        'fetch_content': lambda contex, config: importlib.import_module('llm_workers.tools.fetch').fetch_content,
        'fetch_page_markdown': lambda contex, config: importlib.import_module('llm_workers.tools.fetch').fetch_page_markdown,
        'fetch_page_text': lambda contex, config: importlib.import_module('llm_workers.tools.fetch').fetch_page_text,
        'fetch_page_links': lambda contex, config: importlib.import_module('llm_workers.tools.fetch').fetch_page_links,
        'whisper_cpp': lambda contex, config: importlib.import_module('llm_workers.tools.transcribe_whisper_cpp').make_transcript,
        't2_ai_wrapper': lambda contex, config: importlib.import_module('llm_workers.tools.t2_ai_wrapper').t2_ai_wrapper,
        'llm': lambda contex, config: llm_workers.tools.llm_tool.build_llm_tool(contex, config),
     }

    def __init__(self, config: WorkerConfig):
        self._config = config
        self._models = dict[str, BaseChatModel]()
        self._tools = dict[str, BaseTool]()
        # register models
        for model in self._config.models:
            model_params = model.model_params or {}
            self._models[model.name] = init_chat_model(model.model, model_provider=model.provider, configurable_fields=None, **model_params)
        # register custom tools
        for definition in self._config.custom_tools:
            try:
                tool = build_custom_tool(definition, self)
            except Exception as e:
                raise ValueError(f"Failed to create custom tool {definition.name}: {e}", e)
            if tool.name in self._tools:
                raise ValueError(f"Failed to create custom tool {definition.name}: tool already defined")
            self._tools[tool.name] = tool

    @property
    def config(self) -> WorkerConfig:
        return self._config

    def _register_tool(self, tool: BaseTool):
        if tool.name in self._tools:
            logger.info(f"Redefining tool {tool.name}")
        self._tools[tool.name] = tool

    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        has_config = len(config) > 0
        if not has_config:
            if tool_name in self._tools:
                return self._tools[tool_name]
            if tool_name in self._predefined_tools_factories:
                try:
                    tool = self._predefined_tools_factories[tool_name](self, config)
                except Exception as e:
                    raise ValueError(f"Failed to create tool {tool_name}: {e}", e)
                self._register_tool(tool)
                return tool
            else:
                available_tools = list(self._tools.keys()) + list(self._predefined_tools_factories.keys())
                available_tools = list(dict.fromkeys(available_tools)) # remove duplicates
                raise ValueError(f"Tool {tool_name} not found, available tools: {available_tools}")
        else:
            if tool_name in self._tools:
                raise ValueError(f"Cannot create tool {tool_name} with custom config, tool already exists")
            if tool_name in self._predefined_tools_factories:
                try:
                    return self._predefined_tools_factories[tool_name](self, config)
                except Exception as e:
                    raise ValueError(f"Failed to create tool {tool_name} with custom config: {e}", e)
            else:
                available_tools = list(self._tools.keys()) + list(self._predefined_tools_factories.keys())
                available_tools = list(dict.fromkeys(available_tools)) # remove duplicates
                raise ValueError(f"Cannot create tool {tool_name} with custom config, known tools: {available_tools}")

    def get_llm(self, llm_name: str):
        if llm_name in self._models:
            return self._models[llm_name]
        raise ValueError(f"LLM {llm_name} not found")