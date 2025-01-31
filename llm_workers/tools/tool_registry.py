import importlib
import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition
from llm_workers.tools.stub_tool import StubToolDefinition, build_stub_tool
from llm_workers.tools.llm_tool import build_llm_tool, LLMToolDefinition
from llm_workers.tools.tool_binding import ToolBindingDefinition, build_tool_binding

logger = logging.getLogger(__name__)


class ToolRegistry:

    # Factory methods for predefined tools, used for resolving tool references
    # with dependencies only if the tool is actually used
    _predefined_tools_factories = {
        'fetch_page_text': lambda x: importlib.import_module('llm_workers.tools.fetch_tools').fetch_page_text,
        'fetch_page_links': lambda x: importlib.import_module('llm_workers.tools.fetch_tools').fetch_page_links,
        'whisper_cpp': lambda x: importlib.import_module('llm_workers.tools.transcribe_whisper_cpp').make_transcript,
    }

    def __init__(self):
        # hard-coded tools
        self.tools = {}

    def register_custom_tools(self, models_lookup: Callable[[str], BaseChatModel], definitions: list[CustomToolBaseDefinition]):
        for definition in definitions:
            self._register_tool(self._build_tool(models_lookup, definition))

    def _build_tool(self, models_lookup: Callable[[str], BaseChatModel], definition: CustomToolBaseDefinition) -> BaseTool:
        try:
            if isinstance(definition, StubToolDefinition):
                return build_stub_tool(definition)
            if isinstance(definition, LLMToolDefinition):
                return build_llm_tool(definition, models_lookup, self.resolve_tool_refs)
            if isinstance(definition, ToolBindingDefinition):
                return build_tool_binding(definition, self.resolve_tool_refs)
            raise ValueError(f"Unsupported custom tool definition class {type(definition)}")
        except Exception as e:
            raise ValueError(f"Error building custom tool {definition.name}") from e

    def _register_tool(self, tool: BaseTool):
        if tool.name in self.tools:
            logger.info(f"Redefining tool {tool.name}")
        self.tools[tool.name] = tool

    def _get_tool(self, name: str) -> BaseTool:
        if name in self.tools:
            return self.tools[name]
        else:
            if name in self._predefined_tools_factories:
                tool = self._predefined_tools_factories[name](name)
                self._register_tool(tool)
                return tool
            else:
                available_tools = list(self.tools.keys()) + list(self._predefined_tools_factories.keys())
                available_tools = list(dict.fromkeys(available_tools)) # remove duplicates
                raise ValueError(f"Tool {name} not found, available tools: {available_tools}")

    def resolve_tool_refs(self, tool_refs: list[str]) -> list[BaseTool]:
        return [self._get_tool(tool_ref) for tool_ref in tool_refs]
