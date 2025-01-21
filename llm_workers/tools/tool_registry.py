from langchain_core.tools import Tool

from llm_workers.tools.metacritic_monkey_tool import metacritic_monkey_tool


class ToolRegistry:
    def __init__(self):
        self.tools = {
            "metacritic_monkey": metacritic_monkey_tool
        }

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str):
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found in registry, available tools: {self.tools.keys()}")
        return self.tools[name]

    def resolve_tool_refs(self, tool_refs: list[str]) -> list[Tool]:
        return [self.get_tool(tool_ref) for tool_ref in tool_refs]
