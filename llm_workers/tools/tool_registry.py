from langchain_core.tools import BaseTool

from llm_workers.tools.custom_tools_base import CustomToolBaseConfig
from llm_workers.tools.debug_tool import DebugToolConfig, build_debug_tool
from llm_workers.tools.llm_tool import build_llm_tool, LLMToolConfig


class ToolRegistry:
    def __init__(self):
        # hard-coded tools
        self.tools = {
        }

    def register_custom_tools(self, tool_models: list[CustomToolBaseConfig]):
        for tool_model in tool_models:
            self._register_tool(self._build_tool(tool_model))

    def _build_tool(self, tool_model: CustomToolBaseConfig) -> BaseTool:
        if isinstance(tool_model, DebugToolConfig):
            return build_debug_tool(tool_model)
        if isinstance(tool_model, LLMToolConfig):
            tool_lookup = lambda tool_refs: self.resolve_tool_refs(tool_refs)
            return build_llm_tool(tool_model, tool_lookup)

    def _register_tool(self, tool: BaseTool):
        if tool.name in self.tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self.tools[tool.name] = tool

    def _get_tool(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found, available tools: {self.tools.keys()}")
        return self.tools[name]

    def resolve_tool_refs(self, tool_refs: list[str]) -> list[BaseTool]:
        return [self._get_tool(tool_ref) for tool_ref in tool_refs]
