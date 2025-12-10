import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from llm_workers.core.token_tracking import SimpleTokenUsageTracker, CompositeTokenUsageTracker, _extract_usage_metadata_from_message


class TestSimpleTokenUsageTracker(unittest.TestCase):
    """Test cases for SimpleTokenUsageTracker."""

    def setUp(self):
        self.tracker = SimpleTokenUsageTracker()

    def test_initial_state(self):
        """Test that tracker starts with zero tokens."""
        self.assertEqual(self.tracker.total_tokens, 0)
        self.assertEqual(self.tracker.input_tokens, 0)
        self.assertEqual(self.tracker.output_tokens, 0)
        self.assertEqual(self.tracker.reasoning_tokens, 0)
        self.assertEqual(self.tracker.cache_read_tokens, 0)

    def test_update_from_metadata_basic(self):
        """Test updating from metadata dictionary."""
        metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        self.tracker.update_from_metadata(metadata)

        self.assertEqual(self.tracker.total_tokens, 100)
        self.assertEqual(self.tracker.input_tokens, 60)
        self.assertEqual(self.tracker.output_tokens, 40)

    def test_update_from_metadata_with_details(self):
        """Test updating from metadata with reasoning and cache details."""
        metadata = {
            'total_tokens': 200,
            'input_tokens': 120,
            'output_tokens': 80,
            'output_token_details': {'reasoning': 30},
            'input_token_details': {'cache_read': 20}
        }

        self.tracker.update_from_metadata(metadata)

        self.assertEqual(self.tracker.total_tokens, 200)
        self.assertEqual(self.tracker.input_tokens, 120)
        self.assertEqual(self.tracker.output_tokens, 80)
        self.assertEqual(self.tracker.reasoning_tokens, 30)
        self.assertEqual(self.tracker.cache_read_tokens, 20)

    def test_accumulation(self):
        """Test that tokens accumulate correctly across multiple updates."""
        metadata1 = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        metadata2 = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }

        self.tracker.update_from_metadata(metadata1)
        self.tracker.update_from_metadata(metadata2)

        self.assertEqual(self.tracker.total_tokens, 150)
        self.assertEqual(self.tracker.input_tokens, 90)
        self.assertEqual(self.tracker.output_tokens, 60)

    def test_reset(self):
        """Test that reset clears all counters."""
        metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        self.tracker.update_from_metadata(metadata)
        self.tracker.reset()

        self.assertEqual(self.tracker.total_tokens, 0)
        self.assertEqual(self.tracker.input_tokens, 0)
        self.assertEqual(self.tracker.output_tokens, 0)

    def test_format_usage_zero(self):
        """Test formatting when no tokens are used."""
        self.assertIsNone(self.tracker.format_usage())

    def test_format_usage_basic(self):
        """Test basic token usage formatting."""
        self.tracker.total_tokens = 100
        self.tracker.input_tokens = 60
        self.tracker.output_tokens = 40

        result = self.tracker.format_usage()
        self.assertEqual(result, "100 (60 in, 40 out)")

    def test_format_usage_with_reasoning(self):
        """Test formatting with reasoning tokens."""
        self.tracker.total_tokens = 150
        self.tracker.input_tokens = 60
        self.tracker.output_tokens = 90
        self.tracker.reasoning_tokens = 30

        result = self.tracker.format_usage()
        self.assertEqual(result, "150 (60 in, 90 out, 30 reasoning)")

    def test_format_usage_with_cache(self):
        """Test formatting with cache tokens."""
        self.tracker.total_tokens = 100
        self.tracker.input_tokens = 60
        self.tracker.output_tokens = 40
        self.tracker.cache_read_tokens = 20

        result = self.tracker.format_usage()
        self.assertEqual(result, "100 (60 in, 40 out) | Cache: 20")


class TestCompositeTokenUsageTracker(unittest.TestCase):
    """Test cases for CompositeTokenUsageTracker."""

    def setUp(self):
        self.tracker = CompositeTokenUsageTracker()

    def test_initial_state(self):
        """Test that tracker starts with zero tokens."""
        self.assertIsNone(self.tracker.format_current_usage())
        self.assertIsNone(self.tracker.format_total_usage())

    def test_single_model_update(self):
        """Test updating with a single model."""
        message = Mock(spec=AIMessage)
        message.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        self.tracker.update_from_message(message, "default")

        # Current usage shows session tokens
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 100 (60 in, 40 out)")

        # Total usage shows per-model breakdown
        expected_total = "Total Session Tokens: 100 total\n  default: 100 (60 in, 40 out)"
        self.assertEqual(self.tracker.format_total_usage(), expected_total)

    def test_multiple_model_updates(self):
        """Test updating with multiple different models."""
        # Add tokens for 'default' model
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }
        self.tracker.update_from_message(message1, "default")

        # Add tokens for 'fast' model
        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }
        self.tracker.update_from_message(message2, "fast")

        # Current usage shows combined session tokens
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 150 (90 in, 60 out)")

        # Total usage shows per-model breakdown
        expected_total = ("Total Session Tokens: 150 total\n"
                         "  default: 100 (60 in, 40 out)\n"
                         "  fast: 50 (30 in, 20 out)")
        self.assertEqual(self.tracker.format_total_usage(), expected_total)

    def test_same_model_multiple_updates(self):
        """Test multiple updates to the same model accumulate correctly."""
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }

        self.tracker.update_from_message(message1, "default")
        self.tracker.update_from_message(message2, "default")

        # Current usage shows combined session tokens
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 150 (90 in, 60 out)")

        # Total usage shows accumulated per-model tokens
        expected_total = "Total Session Tokens: 150 total\n  default: 150 (90 in, 60 out)"
        self.assertEqual(self.tracker.format_total_usage(), expected_total)


    def test_reasoning_tokens_handling(self):
        """Test handling of reasoning tokens in both current and total usage."""
        message = Mock(spec=AIMessage)

        # Mock hasattr behavior
        import builtins
        original_hasattr = builtins.hasattr
        def mock_hasattr(obj, name):
            if name == 'usage_metadata':
                return False
            if name == 'response_metadata':
                return True
            return False
        builtins.hasattr = mock_hasattr

        try:
            message.response_metadata = {
                'usage_metadata': {
                    'total_tokens': 200,
                    'input_tokens': 100,
                    'output_tokens': 100,
                    'output_token_details': {'reasoning': 40}
                }
            }

            self.tracker.update_from_message(message, "thinking")

            # Current usage shows reasoning tokens
            self.assertEqual(self.tracker.format_current_usage(), "Tokens: 200 (100 in, 100 out, 40 reasoning)")

            # Total usage shows reasoning tokens in per-model breakdown
            expected_total = "Total Session Tokens: 200 total\n  thinking: 200 (100 in, 100 out, 40 reasoning)"
            self.assertEqual(self.tracker.format_total_usage(), expected_total)
        finally:
            # Restore original hasattr
            builtins.hasattr = original_hasattr

    def test_current_usage_auto_reset(self):
        """Test that current usage automatically resets after format_current_usage is called."""
        # Add initial tokens
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }
        self.tracker.update_from_message(message1, "default")

        # Format current usage (this should reset it)
        current_usage = self.tracker.format_current_usage()
        self.assertEqual(current_usage, "Tokens: 100 (60 in, 40 out)")

        # Current usage should now be reset
        self.assertIsNone(self.tracker.format_current_usage())

        # But total usage should still be preserved
        expected_total = "Total Session Tokens: 100 total\n  default: 100 (60 in, 40 out)"
        self.assertEqual(self.tracker.format_total_usage(), expected_total)

        # Add more tokens
        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }
        self.tracker.update_from_message(message2, "default")

        # Current usage should show only the new tokens
        current_usage = self.tracker.format_current_usage()
        self.assertEqual(current_usage, "Tokens: 50 (30 in, 20 out)")

        # Total usage should show accumulated tokens
        expected_total = "Total Session Tokens: 150 total\n  default: 150 (90 in, 60 out)"
        self.assertEqual(self.tracker.format_total_usage(), expected_total)


class TestTokenTrackingWithTools(unittest.TestCase):
    """Test cases for token tracking with tool messages and additional_kwargs."""

    def setUp(self):
        self.simple_tracker = SimpleTokenUsageTracker()
        self.composite_tracker = CompositeTokenUsageTracker()

    def test_simple_tracker_with_metadata(self):
        """Test SimpleTokenUsageTracker with token usage metadata."""
        metadata = {
            'total_tokens': 150,
            'input_tokens': 90,
            'output_tokens': 60,
            'output_token_details': {'reasoning': 20}
        }

        self.simple_tracker.update_from_metadata(metadata)

        self.assertEqual(self.simple_tracker.total_tokens, 150)
        self.assertEqual(self.simple_tracker.input_tokens, 90)
        self.assertEqual(self.simple_tracker.output_tokens, 60)
        self.assertEqual(self.simple_tracker.reasoning_tokens, 20)


    def test_composite_tracker_with_tool_messages(self):
        """Test CompositeTokenUsageTracker with ToolMessage instances containing usage metadata."""
        tool_message = Mock(spec=ToolMessage)
        tool_message.additional_kwargs = {
            'usage_metadata_per_model': {
                'tool-model': {
                    'total_tokens': 75,
                    'input_tokens': 45,
                    'output_tokens': 30
                }
            }
        }

        ai_message = Mock(spec=AIMessage)
        ai_message.usage_metadata = {
            'total_tokens': 125,
            'input_tokens': 75,
            'output_tokens': 50
        }

        # Update with tool message
        self.composite_tracker.update_from_message(tool_message, "default")

        # Update with AI message
        self.composite_tracker.update_from_message(ai_message, "main-model")

        # Current usage should show combined tokens
        current_usage = self.composite_tracker.format_current_usage()
        self.assertIn("200", current_usage)  # 75 + 125 total tokens
        self.assertIn("120 in", current_usage)  # 45 + 75 input tokens
        self.assertIn("80 out", current_usage)  # 30 + 50 output tokens

        # Total usage should show per-model breakdown
        total_usage = self.composite_tracker.format_total_usage()
        self.assertIn("Total Session Tokens: 200 total", total_usage)
        self.assertIn("main-model: 125 (75 in, 50 out)", total_usage)
        self.assertIn("tool-model: 75 (45 in, 30 out)", total_usage)

    def test_attach_usage_to_message(self):
        """Test the attach_usage_to_message method attaches tracker's internal state."""
        # First, populate the tracker with some usage data
        ai_message = Mock(spec=AIMessage)
        ai_message.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }
        self.composite_tracker.update_from_message(ai_message, "test-model")

        # Create a tool message and attach the tracker's usage data
        tool_message = ToolMessage(content="test", tool_call_id="123", name="test_tool")
        self.composite_tracker.attach_usage_to_message(tool_message)

        # Verify the metadata was attached with per-model format
        self.assertIn('usage_metadata', tool_message.additional_kwargs)
        self.assertIn('test-model', tool_message.additional_kwargs['usage_metadata'])
        self.assertEqual(tool_message.additional_kwargs['usage_metadata']['test-model']['total_tokens'], 50)
        self.assertEqual(tool_message.additional_kwargs['usage_metadata']['test-model']['input_tokens'], 30)
        self.assertEqual(tool_message.additional_kwargs['usage_metadata']['test-model']['output_tokens'], 20)

    def test_message_without_metadata(self):
        """Test that messages without metadata are handled gracefully."""
        message = Mock(spec=ToolMessage)
        # Don't set any metadata attributes

        # Mock hasattr to return False for all metadata attributes
        import builtins
        original_hasattr = builtins.hasattr
        def mock_hasattr(obj, name):
            return False
        builtins.hasattr = mock_hasattr

        try:
            # This should not raise an error and should not update any tokens
            self.composite_tracker.update_from_message(message, "default")

            # No tokens should be tracked
            self.assertIsNone(self.composite_tracker.format_current_usage())
        finally:
            builtins.hasattr = original_hasattr


    def test_simple_tracker_update_from_usage_metadata(self):
        """Test SimpleTokenUsageTracker.update_from_usage_metadata method."""
        usage_metadata = {
            'total_tokens': 200,
            'input_tokens': 120,
            'output_tokens': 80,
            'output_token_details': {'reasoning': 30},
            'input_token_details': {'cache_read': 25}
        }

        self.simple_tracker.update_from_metadata(usage_metadata)

        self.assertEqual(self.simple_tracker.total_tokens, 200)
        self.assertEqual(self.simple_tracker.input_tokens, 120)
        self.assertEqual(self.simple_tracker.output_tokens, 80)
        self.assertEqual(self.simple_tracker.reasoning_tokens, 30)
        self.assertEqual(self.simple_tracker.cache_read_tokens, 25)

    def test_composite_tracker_update_from_metadata(self):
        """Test CompositeTokenUsageTracker.update_from_metadata method with per-model dict."""
        usage_metadata_per_model = {
            'test-model': {
                'total_tokens': 150,
                'input_tokens': 90,
                'output_tokens': 60,
                'output_token_details': {'reasoning': 20}
            }
        }

        self.composite_tracker.update_from_metadata(usage_metadata_per_model)

        # Check current usage
        current_usage = self.composite_tracker.format_current_usage()
        self.assertIn("150", current_usage)
        self.assertIn("90 in", current_usage)
        self.assertIn("60 out", current_usage)
        self.assertIn("20 reasoning", current_usage)

        # Check per-model total usage
        total_usage = self.composite_tracker.format_total_usage()
        self.assertIn("test-model: 150 (90 in, 60 out, 20 reasoning)", total_usage)


class TestExtractUsageMetadata(unittest.TestCase):
    """Test cases for the module-level extract_usage_metadata_from_message function."""

    def test_extract_from_additional_kwargs(self):
        """Test extracting metadata from additional_kwargs with usage_metadata_per_model."""
        message = Mock(spec=ToolMessage)
        expected_metadata = {
            'test-model': {
                'total_tokens': 100,
                'input_tokens': 60,
                'output_tokens': 40
            }
        }
        message.additional_kwargs = {'usage_metadata_per_model': expected_metadata}

        result = _extract_usage_metadata_from_message(message, 'default')
        self.assertEqual(result, expected_metadata)

    def test_extract_from_usage_metadata_attribute(self):
        """Test extracting metadata from usage_metadata attribute."""
        message = Mock(spec=AIMessage)
        metadata = {
            'total_tokens': 120,
            'input_tokens': 70,
            'output_tokens': 50
        }
        message.usage_metadata = metadata

        result = _extract_usage_metadata_from_message(message, 'default')
        # Should wrap in per-model dict with default model name
        self.assertEqual(result, {'default': metadata})

    def test_extract_from_response_metadata(self):
        """Test extracting metadata from response_metadata."""
        message = Mock(spec=AIMessage)
        metadata = {
            'total_tokens': 180,
            'input_tokens': 100,
            'output_tokens': 80
        }
        message.response_metadata = {'usage_metadata': metadata}

        # Mock hasattr behavior
        import builtins
        original_hasattr = builtins.hasattr
        def mock_hasattr(obj, name):
            if name == 'additional_kwargs':
                return False
            if name == 'usage_metadata':
                return False
            if name == 'response_metadata':
                return True
            return False
        builtins.hasattr = mock_hasattr

        try:
            result = _extract_usage_metadata_from_message(message, 'my-model')
            # Should wrap in per-model dict with provided model name
            self.assertEqual(result, {'my-model': metadata})
        finally:
            builtins.hasattr = original_hasattr

    def test_extract_priority_additional_kwargs_first(self):
        """Test that additional_kwargs usage_metadata_per_model takes priority over other sources."""
        message = Mock(spec=AIMessage)

        priority_metadata = {'model-a': {'total_tokens': 100, 'source': 'additional_kwargs'}}
        other_metadata = {'total_tokens': 200, 'source': 'usage_metadata'}

        message.additional_kwargs = {'usage_metadata_per_model': priority_metadata}
        message.usage_metadata = other_metadata

        result = _extract_usage_metadata_from_message(message, 'default')
        self.assertEqual(result, priority_metadata)
        self.assertEqual(result['model-a']['source'], 'additional_kwargs')

    def test_extract_returns_none_when_no_metadata(self):
        """Test that function returns None when no metadata is found."""
        message = Mock(spec=ToolMessage)

        # Mock hasattr to return False for all attributes
        import builtins
        original_hasattr = builtins.hasattr
        def mock_hasattr(obj, name):
            return False
        builtins.hasattr = mock_hasattr

        try:
            result = _extract_usage_metadata_from_message(message, 'default')
            self.assertIsNone(result)
        finally:
            builtins.hasattr = original_hasattr


if __name__ == '__main__':
    unittest.main()