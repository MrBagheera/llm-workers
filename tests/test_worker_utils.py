import unittest
from unittest.mock import Mock
from typing import Generator, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException

from llm_workers.api import WorkerNotification
from llm_workers.config import ToolDefinition
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.worker_utils import call_tool


class TestCallToolConvertExceptions(unittest.TestCase):
    """Tests for the convert_tool_exceptions parameter in call_tool function."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create a mock evaluation context
        self.eval_context = EvaluationContext()

        # Create a mock token tracker
        self.token_tracker = CompositeTokenUsageTracker()

        # Create a basic config
        self.config = RunnableConfig()

    def test_tool_exception_not_converted_by_default(self):
        """Test that ToolException is raised by default when convert_tool_exceptions=False."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("Tool failed deliberately")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='failing_tool'
        )}

        input_data = {'input_str': 'test input'}

        # Call with convert_tool_exceptions=False (default)
        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=False
        )

        # Should raise ToolException
        with self.assertRaises(ToolException) as context:
            list(generator)  # Consume generator to trigger execution

        self.assertIn("Tool failed deliberately", str(context.exception))

    def test_tool_exception_converted_to_string(self):
        """Test that ToolException is converted to string result when convert_tool_exceptions=True."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("Tool failed deliberately")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(name='failing_tool')}

        input_data = {'input_str': 'test input'}

        # Call with convert_tool_exceptions=True
        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        # Collect notifications and result
        notifications = []
        result = None
        try:
            while True:
                chunk = next(generator)
                if isinstance(chunk, WorkerNotification):
                    notifications.append(chunk)
        except StopIteration as e:
            result = e.value

        # Should not raise, result should be an error string
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertIn("Tool Error:", result)
        self.assertIn("Tool failed deliberately", result)

    def test_successful_tool_execution_unaffected(self):
        """Test that successful tool execution is unaffected by convert_tool_exceptions parameter."""

        class SuccessfulTool(BaseTool):
            name: str = "successful_tool"
            description: str = "A tool that always succeeds"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Success: {input_str}"

        tool = SuccessfulTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='successful_tool'
        )}

        input_data = {'input_str': 'test input'}

        # Test with convert_tool_exceptions=False
        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=False
        )

        notifications = []
        result = None
        try:
            while True:
                chunk = next(generator)
                if isinstance(chunk, WorkerNotification):
                    notifications.append(chunk)
        except StopIteration as e:
            result = e.value

        self.assertEqual(result, "Success: test input")

        # Test with convert_tool_exceptions=True
        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        notifications = []
        result = None
        try:
            while True:
                chunk = next(generator)
                if isinstance(chunk, WorkerNotification):
                    notifications.append(chunk)
        except StopIteration as e:
            result = e.value

        # Result should be identical
        self.assertEqual(result, "Success: test input")

    def test_non_tool_exception_not_converted(self):
        """Test that non-ToolException errors are not converted, even with convert_tool_exceptions=True."""

        class CrashingTool(BaseTool):
            name: str = "crashing_tool"
            description: str = "A tool that crashes with non-ToolException"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ValueError("Unexpected error")

        tool = CrashingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='crashing_tool'
        )}

        input_data = {'input_str': 'test input'}

        # Call with convert_tool_exceptions=True
        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        # Should still raise ValueError, not convert it
        with self.assertRaises(ValueError) as context:
            list(generator)

        self.assertIn("Unexpected error", str(context.exception))

    def test_tool_exception_message_format(self):
        """Test the exact format of the error message when converting ToolException."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("Custom error message with details")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='failing_tool'
        )}

        input_data = {'input_str': 'test input'}

        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        result = None
        try:
            while True:
                next(generator)
        except StopIteration as e:
            result = e.value

        # Check exact format
        self.assertEqual(result, "Tool Error: Custom error message with details")

    def test_notifications_still_yielded_on_exception(self):
        """Test that tool_start and tool_end notifications are still yielded when exception is converted."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("Tool failed")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='failing_tool'
        )}

        input_data = {'input_str': 'test input'}

        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True,
            ui_hint_override=None
        )

        notifications = []
        result = None
        try:
            while True:
                chunk = next(generator)
                if isinstance(chunk, WorkerNotification):
                    notifications.append(chunk)
        except StopIteration as e:
            result = e.value

        # Should have tool_start and tool_end notifications
        self.assertEqual(len(notifications), 2)
        self.assertEqual(notifications[0].type, 'tool_start')
        self.assertEqual(notifications[1].type, 'tool_end')

        # Result should be error string
        self.assertIn("Tool Error:", result)

    def test_empty_tool_exception_message(self):
        """Test handling of ToolException with empty message."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='failing_tool'
        )}

        input_data = {'input_str': 'test input'}

        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        result = None
        try:
            while True:
                next(generator)
        except StopIteration as e:
            result = e.value

        # Should still have "Tool Error:" prefix
        self.assertEqual(result, "Tool Error: ")

    def test_tool_exception_with_special_characters(self):
        """Test that special characters in exception message are preserved."""

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that fails"

            def _run(self, input_str: str, **kwargs) -> str:
                raise ToolException("Error with 'quotes', \"double quotes\", and\nnewlines\ttabs")

        tool = FailingTool()
        tool.metadata = {'tool_definition': ToolDefinition(
            name='failing_tool'
        )}

        input_data = {'input_str': 'test input'}

        generator = call_tool(
            tool=tool,
            input=input_data,
            evaluation_context=self.eval_context,
            token_tracker=self.token_tracker,
            config=self.config,
            kwargs={},
            convert_tool_exceptions=True
        )

        result = None
        try:
            while True:
                next(generator)
        except StopIteration as e:
            result = e.value

        # All special characters should be preserved
        self.assertIn("'quotes'", result)
        self.assertIn('"double quotes"', result)
        self.assertIn("\n", result)
        self.assertIn("\t", result)


if __name__ == "__main__":
    unittest.main()
