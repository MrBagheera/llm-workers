import importlib
import inspect
import logging
from copy import copy
from typing import Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.rate_limiters import InMemoryRateLimiter

from llm_workers.api import WorkersContext, WorkerException, ExtendedBaseTool
from llm_workers.config import WorkersConfig, load_config, StandardModelDefinition, ImportModelDefinition, \
    ToolDefinition, ToolReference
from llm_workers.tools.custom_tool import build_custom_tool

logger = logging.getLogger(__name__)


class StandardContext(WorkersContext):

    def __init__(self, config: WorkersConfig):
        self._config = config
        self._models = dict[str, BaseChatModel]()
        self._tools = dict[str, BaseTool]()
        self._register_models()
        self._register_tools()

    def _register_models(self):
        # register models
        for model_def in self._config.models:
            model_params = copy(model_def.config) if model_def.config else model_def.model_extra
            if model_def.rate_limiter:
                model_params['rate_limiter'] = InMemoryRateLimiter(
                    requests_per_second = model_def.rate_limiter.requests_per_second,
                    check_every_n_seconds = model_def.rate_limiter.check_every_n_seconds,
                    max_bucket_size = model_def.rate_limiter.max_bucket_size)
            model: BaseChatModel
            try:
                if isinstance(model_def, StandardModelDefinition):
                    model = init_chat_model(model_def.model, model_provider=model_def.provider,
                                            configurable_fields=None, **model_params)
                elif isinstance(model_def, ImportModelDefinition):
                    # split model.import_from into module_name and symbol
                    segments = model_def.import_from.split('.')
                    module_name = '.'.join(segments[:-1])
                    symbol_name = segments[-1]
                    module = importlib.import_module(module_name)  # Import the module
                    symbol = getattr(module, symbol_name, None)  # Retrieve the symbol
                    if symbol is None:
                        raise ValueError(f"Cannot import model from {model_def.import_from}: symbol {symbol_name} not found")
                    elif isinstance(symbol, BaseChatModel):
                        model = symbol
                    elif inspect.isclass(symbol):
                        model = symbol(**model_params) # use default constructor
                    elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
                        model = symbol(**model_params) # use default constructor
                    else:
                        raise ValueError(f"Invalid symbol type {type(symbol)}")
                    if not isinstance(model, BaseChatModel):
                        raise ValueError(f"Invalid model type {type(model)}")
                else:
                    raise ValueError(f"Invalid config type {type(model_def)}")
            except Exception as e:
                raise WorkerException(f"Failed to create model {model_def.name}: {e}", e)

            self._models[model_def.name] = model
            logger.info(f"Registered model {model_def.name}")

    # noinspection DuplicatedCode
    def _register_tools(self):
        for tool_def in self._config.tools:
            if tool_def.name in self._tools:
                raise WorkerException(f"Failed to create tool {tool_def.name}: tool already defined")
            tool = self._create_tool(tool_def)
            self._tools[tool.name] = tool
            logger.info(f"Registered tool {tool.name}")

    def _create_tool(self, tool_def: ToolDefinition) -> BaseTool:
        try:
            if tool_def.import_from is not None:
                tool = self._import_tool(tool_def)
            else:
                tool = build_custom_tool(tool_def, self)
            # common post-processing
            if tool_def.return_direct is not None:
                tool.return_direct = tool_def.return_direct
            if tool_def.confidential:   # confidential implies return_direct
                tool.return_direct = True
            if tool.metadata is None:
                tool.metadata = {}
            tool.metadata['tool_definition'] = tool_def
            if isinstance(tool, ExtendedBaseTool):
                tool.metadata['__extension'] = tool # TODO really hackish
            return tool
        except ImportError as e:
            raise WorkerException(f"Failed to import module for tool {tool_def.name}: {e}")
        except Exception as e:
            raise WorkerException(f"Failed to create tool {tool_def.name}: {e}", e)

    def _import_tool(self, tool_def: ToolDefinition) -> BaseTool:
        tool_config = copy(tool_def.config if tool_def.config else tool_def.model_extra)
        tool_config['name'] = tool_def.name
        if tool_def.description is not None:
            tool_config['description'] = tool_def.description
        # split model.import_from into module_name and symbol
        segments = tool_def.import_from.split('.')
        module_name = '.'.join(segments[:-1])
        symbol_name = segments[-1]
        module = importlib.import_module(module_name)  # Import the module
        symbol = getattr(module, symbol_name)  # Retrieve the symbol
        # make the tool
        if symbol is None:
            raise ValueError(f"Cannot import tool from {tool_def.import_from}: symbol {symbol_name} not found")
        elif isinstance(symbol, BaseTool):
            tool = symbol
        elif inspect.isclass(symbol):
            tool = symbol(**tool_config) # use default constructor
        elif inspect.isfunction(symbol) or inspect.ismethod(symbol):
            if len(symbol.__annotations__) >= 2 and 'context' in symbol.__annotations__ and 'tool_config' in symbol.__annotations__:
                tool = symbol(context = self, tool_config = tool_config)
            else:
                raise ValueError("Invalid tool factory signature, must be `def factory(context: WorkersContext, tool_config: dict[str, any]) -> BaseTool`")
        else:
            raise ValueError(f"Invalid symbol type {type(symbol)}")
        if not isinstance(tool, BaseTool):
            raise ValueError(f"Not a BaseTool: {type(tool)}")
        # overrides for un-cooperating tools
        tool.name = tool_def.name
        if tool_def.description is not None:
            tool.description = tool_def.description
        return tool

    @classmethod
    def load(cls, script_name: str):
        logger.info(f"Loading {script_name}")
        return cls(load_config(script_name))

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

    def get_tool(self, tool_ref: ToolReference) -> BaseTool:
        if isinstance(tool_ref, ToolDefinition):
            return self._create_tool(tool_ref)
        if tool_ref in self._tools:
            return self._tools[tool_ref]
        else:
            available_tools = list(self._tools.keys())
            available_tools.sort()
            raise ValueError(f"Tool {tool_ref} not found, available tools: {available_tools}")

    def get_llm(self, llm_name: str):
        if llm_name in self._models:
            return self._models[llm_name]
        raise WorkerException(f"LLM {llm_name} not found")

    def get_start_tool_message(self, tool_name: str, tool_meta: Dict[str, Any], inputs: Dict[str, Any]) -> str | None:
        try:
            # check if ui_hint is defined in tool definition
            tool_def = tool_meta['tool_definition']
            if tool_def.ui_hint_template is not None:
                hint = tool_def.ui_hint_template.format(**inputs)
                if hint.strip():  # only return if hint is not empty
                    return hint
                else:
                    return None  # empty hint means no message should be shown
            # fallback to ExtendedBaseTool
            if '__extension' in tool_meta:
                extension: ExtendedBaseTool = tool_meta['__extension']
                hint = extension.get_ui_hint(inputs)
                if hint.strip():  # only return if hint is not empty
                    return hint
                else:
                    return None  # empty hint means no message should be shown
        except Exception as e:
            logger.warning(f"Unexpected exception formating start message for tool {tool_name}: {e}", exc_info=True)
        # default
        return f"Running tool {tool_name}"
