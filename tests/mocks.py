from typing import List, Optional, Any, Dict
from unittest.mock import Mock

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from llm_workers.api import WorkersContext, UserContext
from llm_workers.config import WorkersConfig, ToolsDefinitionOrReference
from llm_workers.expressions import EvaluationContext, JsonExpression
from llm_workers.user_context import UserContext
from llm_workers.workers_context import StandardWorkersContext


class MockInvokeLLM(BaseChatModel):
    """Mock LLM that expects invoke() calls with predefined expectations."""

    def __init__(self):
        super().__init__()
        self._expectations: List[tuple[List[BaseMessage], AIMessage]] = []
        self._call_count: int = 0

    def expect_invoke(self, input: List[BaseMessage], output: AIMessage) -> "MockInvokeLLM":
        """Add expectation for the next invoke() call."""
        self._expectations.append((input, output))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Required by BaseChatModel but not used in our invoke path."""
        raise NotImplementedError("Use invoke() instead")

    def invoke(
            self,
            input: List[BaseMessage],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ) -> AIMessage:
        """Validate input and return the pre-configured output message."""
        if self._call_count >= len(self._expectations):
            raise AssertionError(f"Unexpected invoke() call #{self._call_count + 1}. Only {len(self._expectations)} calls were expected.")

        expected_input, output = self._expectations[self._call_count]
        self._call_count += 1

        # Validate input matches expectation
        if len(input) != len(expected_input):
            raise AssertionError(f"invoke() call #{self._call_count}: expected {len(expected_input)} messages, got {len(input)}")

        for i, (expected_msg, actual_msg) in enumerate(zip(expected_input, input)):
            if type(expected_msg) != type(actual_msg):
                raise AssertionError(f"invoke() call #{self._call_count}, message #{i}: expected {type(expected_msg).__name__}, got {type(actual_msg).__name__}")
            if expected_msg.content != actual_msg.content:
                raise AssertionError(f"invoke() call #{self._call_count}, message #{i}: expected content '{expected_msg.content}', got '{actual_msg.content}'")

        return output

    def stream(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any):
        """Stream is not supported for FakeInvokeLLM."""
        raise AssertionError("stream() was called but invoke() was expected")

    def bind_tools(self, tools):
        """Return self for method chaining, no actual tool binding needed."""
        return self

    @property
    def _llm_type(self) -> str:
        return "fake_invoke"

    def verify_all_called(self):
        """Verify that all expected calls were made."""
        if self._call_count < len(self._expectations):
            raise AssertionError(f"Expected {len(self._expectations)} invoke() calls, but only {self._call_count} were made")


class MockStreamLLM(BaseChatModel):
    """Mock LLM that expects stream() calls with predefined expectations."""

    def __init__(self):
        super().__init__()
        self._expectations: List[tuple[List[BaseMessage], List[AIMessageChunk]]] = []
        self._call_count: int = 0

    def expect_stream(self, input: List[BaseMessage], output: List[AIMessageChunk]) -> "MockStreamLLM":
        """Add expectation for the next stream() call."""
        self._expectations.append((input, output))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Required by BaseChatModel but not used in our stream path."""
        raise NotImplementedError("Use stream() instead")

    def invoke(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> AIMessage:
        """Invoke is not supported for FakeStreamLLM."""
        raise AssertionError("invoke() was called but stream() was expected")

    def stream(
            self,
            input: List[BaseMessage],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any
    ):
        """Validate input and stream the pre-configured output chunks."""
        if self._call_count >= len(self._expectations):
            raise AssertionError(f"Unexpected stream() call #{self._call_count + 1}. Only {len(self._expectations)} calls were expected.")

        expected_input, output_chunks = self._expectations[self._call_count]
        self._call_count += 1

        # Validate input matches expectation
        if len(input) != len(expected_input):
            raise AssertionError(f"stream() call #{self._call_count}: expected {len(expected_input)} messages, got {len(input)}")

        for i, (expected_msg, actual_msg) in enumerate(zip(expected_input, input)):
            if type(expected_msg) != type(actual_msg):
                raise AssertionError(f"stream() call #{self._call_count}, message #{i}: expected {type(expected_msg).__name__}, got {type(actual_msg).__name__}")
            if expected_msg.content != actual_msg.content:
                raise AssertionError(f"stream() call #{self._call_count}, message #{i}: expected content '{expected_msg.content}', got '{actual_msg.content}'")

        # Yield chunks
        for chunk in output_chunks:
            yield chunk

    def bind_tools(self, tools):
        """Return self for method chaining, no actual tool binding needed."""
        return self

    @property
    def _llm_type(self) -> str:
        return "fake_stream"

    def verify_all_called(self):
        """Verify that all expected calls were made."""
        if self._call_count < len(self._expectations):
            raise AssertionError(f"Expected {len(self._expectations)} stream() calls, but only {self._call_count} were made")


class StubWorkersContext(WorkersContext):
    """Stub context for testing Worker and custom tools.

    Args:
        llm: Optional BaseChatModel instance for LLM operations
        config: Optional WorkersConfig for configuration
        tools: Optional dictionary mapping tool names to tool instances
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        config: Optional[WorkersConfig] = None,
        tools: Optional[Dict[str, BaseTool]] = None
    ):
        self._llm = llm
        self._config = config or WorkersConfig()
        self._tools = tools or {}
        self._evaluation_context = EvaluationContext()
        for key, expr in self._config.shared.data.items():
            self._evaluation_context.add(key, expr.evaluate(self._evaluation_context))
        self._evaluation_context.mutable = False

    @property
    def config(self) -> WorkersConfig:
        return self._config

    @property
    def evaluation_context(self) -> EvaluationContext:
        return self._evaluation_context

    def get_tool(self, tool_ref, extra_tools: Optional[Dict[str, BaseTool]] = None):
        if extra_tools and tool_ref in extra_tools:
            return extra_tools[tool_ref]
        if tool_ref in self._tools:
            return self._tools[tool_ref]
        raise ValueError(f"Tool {tool_ref} not found in mock context")

    def get_tools(self, scope: str, tool_refs: List[ToolsDefinitionOrReference]) -> List[BaseTool]:
        # for simplicity, we only support string references in this mock
        found_tools = []
        for tool_ref in tool_refs:
            if tool_ref in self._tools:
                found_tools.append(self._tools[tool_ref])
            else:
                raise ValueError(f"Tool {tool_ref} not found in mock context")
        return found_tools

    def get_llm(self, llm_name: str) -> BaseChatModel:
        if self._llm is None:
            raise ValueError("No LLM configured in mock context")
        return self._llm


def create_mock_mcp_tool(name: str, description: str = None) -> BaseTool:
    """Create a simple mock tool for MCP testing.

    Args:
        name: Name of the mock tool
        description: Optional description for the tool

    Returns:
        BaseTool instance that can be used as a mock MCP tool
    """
    desc = description or f"Mock MCP tool: {name}"

    @tool(description=desc)
    def mock_tool(input_str: str) -> str:
        """Mock tool implementation."""
        return f"Result from {name}: {input_str}"

    mock_tool.name = name
    mock_tool.description = desc

    return mock_tool


class MockMCPWorkersContext(StandardWorkersContext):
    """Mock context that simulates MCP tool loading without actual MCP servers.

    This context subclasses StandardWorkersContext and overrides _load_mcp_tools_async
    to return mock tools instead of connecting to real MCP servers. This allows testing
    MCP tool importing without external dependencies.

    Args:
        config: WorkersConfig instance with MCP server definitions
        user_context: UserContext instance for LLM and environment access
        mock_mcp_tools: Dictionary mapping server names to lists of mock tools
    """

    def __init__(self, config: WorkersConfig, user_context: UserContext,
                 mock_mcp_tools: Dict[str, List[BaseTool]] = None):
        super().__init__(config, user_context)
        self._mock_mcp_tools = mock_mcp_tools or {}

    async def _load_mcp_tools_async(self, server_configs: Dict[str, dict]) -> Dict[str, List[BaseTool]]:
        """Override to return mock tools instead of connecting to real MCP servers.

        This method mimics the behavior of the real _load_mcp_tools_async by:
        1. Looking up mock tools for each server
        2. Tagging tools with 'mcp_server' and 'original_name' metadata
        3. Returning a dictionary of tools by server name

        Args:
            server_configs: Dictionary of MCP server configurations

        Returns:
            Dictionary mapping server names to lists of mock tools
        """
        result = {}
        for server_name in server_configs.keys():
            if server_name in self._mock_mcp_tools:
                # Tag tools with server metadata (mimicking real behavior)
                tools = []
                for tool in self._mock_mcp_tools[server_name]:
                    # Create a copy to avoid modifying the original
                    if tool.metadata is None:
                        tool.metadata = {}
                    tool.metadata['mcp_server'] = server_name
                    tool.metadata['original_name'] = tool.name
                    tools.append(tool)
                result[server_name] = tools
            else:
                # Return empty list for unconfigured mock servers
                result[server_name] = []
        return result


def create_context_from_yaml(yaml_str: str,
                              mock_mcp_tools: Dict[str, List[BaseTool]] = None) -> MockMCPWorkersContext:
    """Create a MockMCPWorkersContext from YAML string.

    Args:
        yaml_str: YAML configuration string
        mock_mcp_tools: Optional dictionary mapping MCP server names to lists of mock tools

    Returns:
        MockMCPWorkersContext instance (not yet initialized)
    """
    config_data = yaml.safe_load(yaml_str)
    config = WorkersConfig(**config_data)

    # Create mock user context
    user_context = Mock(spec=UserContext)
    mock_llm = MockInvokeLLM()
    user_context.get_llm = Mock(return_value=mock_llm)

    # Create context with mock MCP tools
    context = MockMCPWorkersContext(config, user_context, mock_mcp_tools)
    return context
