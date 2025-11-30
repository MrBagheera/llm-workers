from typing import List, Optional, Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from llm_workers.api import WorkersContext
from llm_workers.config import WorkersConfig


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

    @property
    def config(self) -> WorkersConfig:
        return self._config

    @property
    def get_public_tools(self) -> List[BaseTool]:
        """Return all tools that don't start with underscore."""
        public_tools = []
        for tool in self._tools.values():
            if not tool.name.startswith("_"):
                public_tools.append(tool)
        return public_tools

    def get_tool(self, tool_ref):
        if tool_ref in self._tools:
            return self._tools[tool_ref]
        raise ValueError(f"Tool {tool_ref} not found in mock context")

    def get_llm(self, llm_name: str) -> BaseChatModel:
        if self._llm is None:
            raise ValueError("No LLM configured in mock context")
        return self._llm

