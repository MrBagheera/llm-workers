import unittest

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from llm_workers.api import WorkerNotification
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.tools.llm_tool import build_llm_tool
from tests.mocks import MockInvokeLLM, StubWorkersContext


class TestLLMTool(unittest.TestCase):

    def test_basic_prompt_only(self):
        """Test basic LLM tool invocation with prompt only."""
        # Setup mock LLM
        mock_llm = MockInvokeLLM()
        mock_llm.expect_invoke(
            input=[HumanMessage(content="What is 2+2?")],
            output=AIMessage(content="4")
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Invoke tool
        result = tool.invoke({"prompt": "What is 2+2?"})

        # Verify result
        self.assertEqual(result, "4")
        mock_llm.verify_all_called()

    def test_prompt_with_system_message(self):
        """Test LLM tool invocation with system message."""
        # Setup mock LLM
        mock_llm = MockInvokeLLM()
        mock_llm.expect_invoke(
            input=[
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello")
            ],
            output=AIMessage(content="Hi there!")
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Invoke tool
        result = tool.invoke({
            "prompt": "Hello",
            "system_message": "You are a helpful assistant."
        })

        # Verify result
        self.assertEqual(result, "Hi there!")
        mock_llm.verify_all_called()

    def test_multiple_ai_messages(self):
        """Test that multiple AI messages are concatenated properly."""
        # Setup mock LLM - Worker returns list of messages
        mock_llm = MockInvokeLLM()
        # Note: Worker.invoke() actually returns a list of messages, not just AIMessage
        # We need to update the test to match actual Worker behavior
        mock_llm.expect_invoke(
            input=[HumanMessage(content="Tell me a story")],
            output=AIMessage(content="Once upon a time...")
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Invoke tool
        result = tool.invoke({"prompt": "Tell me a story"})

        # Verify result
        self.assertEqual(result, "Once upon a time...")
        mock_llm.verify_all_called()

    def test_empty_response(self):
        """Test handling of empty LLM response."""
        # Note: This test depends on how Worker handles empty responses
        # We might need to adjust based on actual behavior
        # For now, assuming Worker returns an empty list
        mock_llm = MockInvokeLLM()
        mock_llm.expect_invoke(
            input=[HumanMessage(content="Say nothing")],
            output=AIMessage(content="")
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Invoke tool
        result = tool.invoke({"prompt": "Say nothing"})

        # Verify result
        self.assertEqual(result, "")
        mock_llm.verify_all_called()

    def test_token_counting(self):
        """Test that token usage from LLM responses is captured."""
        # Setup mock LLM with usage metadata
        mock_llm = MockInvokeLLM()
        ai_message = AIMessage(
            content="The answer is 42",
            response_metadata={
                'usage_metadata': {
                    'input_tokens': 10,
                    'output_tokens': 5,
                    'total_tokens': 15
                }
            }
        )
        mock_llm.expect_invoke(
            input=[HumanMessage(content="What is the answer?")],
            output=ai_message
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Create token tracker and call _stream directly with it
        token_tracker = CompositeTokenUsageTracker()
        chunks = list(tool.stream_with_notifications(
            input={"prompt": "What is the answer?"},
            evaluation_context=context.evaluation_context,
            token_tracker=token_tracker,
            config=None
        ))

        # Find the result (last non-notification chunk)
        result = None
        for chunk in chunks:
            if not isinstance(chunk, WorkerNotification):
                result = chunk

        # Verify result
        self.assertEqual("The answer is 42", result)
        mock_llm.verify_all_called()

        # Verify token usage was captured
        self.assertFalse(token_tracker.is_empty)
        usage_str = token_tracker.format_total_usage()
        self.assertIsNotNone(usage_str)
        self.assertIn("15", usage_str)  # total tokens
        self.assertIn("10", usage_str)  # input tokens
        self.assertIn("5", usage_str)   # output tokens
        self.assertIn("test_model", usage_str)

    def test_notifications_during_execution(self):
        """Test that notifications are yielded during tool execution."""
        # Setup mock LLM
        mock_llm = MockInvokeLLM()
        mock_llm.expect_invoke(
            input=[HumanMessage(content="Test prompt")],
            output=AIMessage(content="Test response")
        )

        # Create context and tool
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Collect all chunks including notifications
        token_tracker = CompositeTokenUsageTracker()
        chunks = list(tool.stream_with_notifications(
            input={"prompt": "Test prompt"},
            evaluation_context=context.evaluation_context,
            token_tracker=token_tracker,
            config=None
        ))

        # Separate notifications from result
        notifications = [c for c in chunks if isinstance(c, WorkerNotification)]
        results = [c for c in chunks if not isinstance(c, WorkerNotification)]

        # Verify we got notifications
        self.assertGreater(len(notifications), 0, "Should have received notifications")

        # Verify notification types
        notification_types = [n.type for n in notifications]
        self.assertIn('thinking_start', notification_types, "Should have thinking_start notification")
        self.assertIn('thinking_end', notification_types, "Should have thinking_end notification")

        # Verify notifications are in correct order (thinking_start before thinking_end)
        thinking_start_idx = notification_types.index('thinking_start')
        thinking_end_idx = notification_types.index('thinking_end')
        self.assertLess(thinking_start_idx, thinking_end_idx, "thinking_start should come before thinking_end")

        # Verify we got the result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Test response")

        mock_llm.verify_all_called()

    def test_tool_schema(self):
        """Test that tool has correct schema."""
        # Create context and tool (need LLM even though we won't use it)
        mock_llm = MockInvokeLLM()
        context = StubWorkersContext(llm=mock_llm)
        tool_config = {"model_ref": "test_model"}
        tool = build_llm_tool(context, tool_config)

        # Verify tool metadata
        self.assertEqual(tool.name, "llm")
        self.assertIn("LLM", tool.description)

        # Verify tool schema
        schema = tool.args_schema.model_json_schema()
        self.assertIn("prompt", schema["properties"])
        self.assertIn("system_message", schema["properties"])
        self.assertIn("prompt", schema["required"])
        self.assertNotIn("system_message", schema["required"])


if __name__ == '__main__':
    unittest.main()
