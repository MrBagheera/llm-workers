import importlib
import logging
from collections.abc import Callable
from typing import Dict, Optional, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, BaseToolkit

import llm_workers.tools.llm_tool
import llm_workers.tools.fetch
from llm_workers.api import WorkersContext, ToolFactory
from llm_workers.config import WorkersConfig, load_config
from llm_workers.tools.custom_tool import build_custom_tool

logger = logging.getLogger(__name__)


class StandardContext(WorkersContext):

    _builtin_tools = [
        llm_workers.tools.fetch.fetch_content,
        llm_workers.tools.fetch.fetch_page_text,
        llm_workers.tools.fetch.fetch_page_links
    ]
    _builtin_tools_factories = {
        'llm': llm_workers.tools.llm_tool.build_llm_tool,
     }

    def __init__(self, config: WorkersConfig):
        self._config = config
        self._models = dict[str, BaseChatModel]()
        self._tool_factories = dict[str, ToolFactory]()
        self._tools = dict[str, BaseTool]()
        self._register_models()
        self._register_builtin_tools_factories()
        self._register_builtin_tools()
        self._register_tools()
        self._register_custom_tools()

    def _register_models(self):
        # register models
        for model in self._config.models:
            model_params = model.model_params or {}
            self._models[model.name] = init_chat_model(model.model, model_provider=model.provider,
                                                       configurable_fields=None, **model_params)

    def _register_builtin_tools_factories(self):
        for tool_name, tool_factory in self._builtin_tools_factories.items():
            self._tool_factories[tool_name] = tool_factory

    def _register_tools(self):
        for tool_def in self._config.tools:
            try:
                segments = tool_def.split('.')
                module = importlib.import_module('.'.join(segments[:-1]))
                symbol = getattr(module, segments[-1])
                if isinstance(symbol, BaseTool):
                    self._register_tool(symbol)
                elif isinstance(symbol, BaseToolkit):
                    for tool_name, tool in symbol.tools.items():
                        self._register_tool(tool)
                elif callable(symbol):
                    if len(symbol.__annotations__) == 2 and 'context' in symbol.__annotations__ and 'config' in symbol.__annotations__:
                        self._tool_factories[symbol.__name__] = symbol
                        logger.info(f"Registered tool factory {symbol.__name__}")
                    else:
                        raise ValueError("invalid tool factory signature")
            except Exception as e:
                raise ValueError(f"Failed to import module {tool_def}: {e}", e)

    def _register_builtin_tools(self):
        for tool in self._builtin_tools:
            self._register_tool(tool)
        pass

    def _register_custom_tools(self):
        # register custom tools
        for definition in self._config.custom_tools:
            try:
                tool = build_custom_tool(definition, self)
            except Exception as e:
                raise ValueError(f"Failed to create custom tool {definition.name}: {e}", e)
            if tool.name in self._tools:
                raise ValueError(f"Failed to create custom tool {definition.name}: tool already defined")
            self._tools[tool.name] = tool
            logger.info(f"Registered tool {tool.name}")

    @classmethod
    def from_file(cls, file_path: str):
        logger.info(f"Loading from {file_path}")
        return cls(load_config(file_path))

    @property
    def config(self) -> WorkersConfig:
        return self._config

    def _register_tool(self, tool: BaseTool):
        redefine = tool.name in self._tools
        self._tools[tool.name] = tool
        if redefine:
            logger.info(f"Redefined tool {tool.name}")
        else:
            logger.info(f"Registered tool {tool.name}")

    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
        if config is None:
            config = {}
        has_config = len(config) > 0
        if not has_config:
            if tool_name in self._tools:
                return self._tools[tool_name]
            if tool_name in self._tool_factories:
                try:
                    tool = self._builtin_tools_factories[tool_name](self, config)
                except Exception as e:
                    raise ValueError(f"Failed to create tool {tool_name}: {e}", e)
                self._register_tool(tool)
                return tool
            else:
                available_tools = list(self._tools.keys()) + list(self._builtin_tools_factories.keys())
                available_tools = list(dict.fromkeys(available_tools)) # remove duplicates
                raise ValueError(f"Tool {tool_name} not found, available tools: {available_tools}")
        else:
            if tool_name in self._builtin_tools_factories:
                try:
                    tool = self._builtin_tools_factories[tool_name](self, config)
                    logger.info(f"Created tool {tool_name}")
                    return tool
                except Exception as e:
                    raise ValueError(f"Failed to create tool {tool_name} with custom config: {e}", e)
            else:
                available_tools = list(self._tools.keys()) + list(self._builtin_tools_factories.keys())
                available_tools = list(dict.fromkeys(available_tools)) # remove duplicates
                raise ValueError(f"Cannot create tool {tool_name} with custom config, known tools: {available_tools}")

    def get_llm(self, llm_name: str):
        if llm_name in self._models:
            return self._models[llm_name]
        raise ValueError(f"LLM {llm_name} not found")