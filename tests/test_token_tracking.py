import unittest
from unittest.mock import Mock
from langchain_core.messages import AIMessage, HumanMessage

from llm_workers.token_tracking import SimpleTokenUsageTracker, CompositeTokenUsageTracker
from llm_workers.config import DisplaySettings, UserConfig


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

    def test_update_from_usage_metadata(self):
        """Test updating from message with usage_metadata attribute."""
        message = Mock(spec=AIMessage)
        message.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        self.tracker.update_from_message(message)

        self.assertEqual(self.tracker.total_tokens, 100)
        self.assertEqual(self.tracker.input_tokens, 60)
        self.assertEqual(self.tracker.output_tokens, 40)

    def test_update_from_response_metadata(self):
        """Test updating from message with response_metadata structure."""
        message = Mock(spec=AIMessage)
        # Configure the mock to return False for usage_metadata hasattr check
        def mock_hasattr(obj, name):
            if name == 'usage_metadata':
                return False
            if name == 'response_metadata':
                return True
            return False

        # Mock hasattr behavior
        import builtins
        original_hasattr = builtins.hasattr
        builtins.hasattr = mock_hasattr

        try:
            message.response_metadata = {
                'usage_metadata': {
                    'total_tokens': 200,
                    'input_tokens': 120,
                    'output_tokens': 80,
                    'output_token_details': {'reasoning': 30},
                    'input_token_details': {'cache_read': 20}
                }
            }

            self.tracker.update_from_message(message)

            self.assertEqual(self.tracker.total_tokens, 200)
            self.assertEqual(self.tracker.input_tokens, 120)
            self.assertEqual(self.tracker.output_tokens, 80)
            self.assertEqual(self.tracker.reasoning_tokens, 30)
            self.assertEqual(self.tracker.cache_read_tokens, 20)
        finally:
            # Restore original hasattr
            builtins.hasattr = original_hasattr

    def test_accumulation(self):
        """Test that tokens accumulate correctly across multiple updates."""
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

        self.tracker.update_from_message(message1)
        self.tracker.update_from_message(message2)

        self.assertEqual(self.tracker.total_tokens, 150)
        self.assertEqual(self.tracker.input_tokens, 90)
        self.assertEqual(self.tracker.output_tokens, 60)

    def test_reset(self):
        """Test that reset clears all counters."""
        message = Mock(spec=AIMessage)
        message.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }

        self.tracker.update_from_message(message)
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

    def test_reset_current_usage_basic(self):
        """Test that reset_current_usage resets session but preserves total per-model."""
        # Add initial tokens
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }
        self.tracker.update_from_message(message1, "default")

        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }
        self.tracker.update_from_message(message2, "fast")

        # Verify initial state
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 150 (90 in, 60 out)")

        # Reset with empty messages
        self.tracker.reset_current_usage([], "default")

        # Current usage should be reset
        self.assertIsNone(self.tracker.format_current_usage())

        # Total usage should still show per-model totals (never reset)
        expected_total = ("Total Session Tokens: 150 total\n"
                         "  default: 100 (60 in, 40 out)\n"
                         "  fast: 50 (30 in, 20 out)")
        self.assertEqual(self.tracker.format_total_usage(), expected_total)

    def test_reset_current_usage_with_messages(self):
        """Test reset_current_usage recalculates from provided messages."""
        # Add initial tokens from multiple models
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }
        self.tracker.update_from_message(message1, "default")

        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 80,
            'input_tokens': 50,
            'output_tokens': 30
        }
        self.tracker.update_from_message(message2, "fast")

        # Create a message list with only one message for recalculation
        remaining_message = Mock(spec=AIMessage)
        remaining_message.usage_metadata = {
            'total_tokens': 30,
            'input_tokens': 20,
            'output_tokens': 10
        }

        messages = [
            HumanMessage(content="test"),
            remaining_message
        ]

        # Reset and recalculate
        self.tracker.reset_current_usage(messages, "default")

        # Current usage should reflect only the remaining message
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 30 (20 in, 10 out)")

        # Total usage should still show original per-model totals
        expected_total = ("Total Session Tokens: 180 total\n"
                         "  default: 100 (60 in, 40 out)\n"
                         "  fast: 80 (50 in, 30 out)")
        self.assertEqual(self.tracker.format_total_usage(), expected_total)

    def test_total_vs_current_independence(self):
        """Test that total and current usage are tracked independently."""
        # Add tokens for default model
        message1 = Mock(spec=AIMessage)
        message1.usage_metadata = {
            'total_tokens': 100,
            'input_tokens': 60,
            'output_tokens': 40
        }
        self.tracker.update_from_message(message1, "default")

        # Reset current (simulating rewind)
        self.tracker.reset_current_usage([], "default")

        # Add more tokens for fast model
        message2 = Mock(spec=AIMessage)
        message2.usage_metadata = {
            'total_tokens': 50,
            'input_tokens': 30,
            'output_tokens': 20
        }
        self.tracker.update_from_message(message2, "fast")

        # Current should only show recent tokens
        self.assertEqual(self.tracker.format_current_usage(), "Tokens: 50 (30 in, 20 out)")

        # Total should show all tokens ever processed
        expected_total = ("Total Session Tokens: 150 total\n"
                         "  default: 100 (60 in, 40 out)\n"
                         "  fast: 50 (30 in, 20 out)")
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


class TestDisplaySettings(unittest.TestCase):
    """Test cases for DisplaySettings configuration."""

    def test_default_values(self):
        """Test that DisplaySettings has correct default values."""
        settings = DisplaySettings()
        self.assertTrue(settings.show_token_usage)
        self.assertFalse(settings.show_reasoning)
        self.assertFalse(settings.auto_open_changed_files)
        self.assertTrue(settings.markdown_output)
        self.assertEqual(settings.file_monitor_include, ['*'])
        self.assertEqual(settings.file_monitor_exclude, ['.*', '*.log'])

    def test_custom_values(self):
        """Test DisplaySettings with custom values."""
        settings = DisplaySettings(
            show_token_usage=False,
            show_reasoning=True,
            auto_open_changed_files=True,
            markdown_output=True,
            file_monitor_include=['*.py', '*.yaml'],
            file_monitor_exclude=['*.pyc', '*.log', '*.tmp']
        )
        self.assertFalse(settings.show_token_usage)
        self.assertTrue(settings.show_reasoning)
        self.assertTrue(settings.auto_open_changed_files)
        self.assertTrue(settings.markdown_output)
        self.assertEqual(settings.file_monitor_include, ['*.py', '*.yaml'])
        self.assertEqual(settings.file_monitor_exclude, ['*.pyc', '*.log', '*.tmp'])


class TestUserConfigDisplaySettings(unittest.TestCase):
    """Test cases for UserConfig with DisplaySettings integration."""

    def test_default_display_settings(self):
        """Test that UserConfig creates default DisplaySettings."""
        config = UserConfig()
        self.assertIsInstance(config.display_settings, DisplaySettings)
        self.assertTrue(config.display_settings.show_token_usage)
        self.assertFalse(config.display_settings.show_reasoning)

    def test_custom_display_settings(self):
        """Test UserConfig with custom DisplaySettings."""
        custom_settings = DisplaySettings(show_reasoning=True, markdown_output=True)
        config = UserConfig(display_settings=custom_settings)
        self.assertTrue(config.display_settings.show_reasoning)
        self.assertTrue(config.display_settings.markdown_output)

    def test_display_settings_from_dict(self):
        """Test UserConfig creation from dictionary with display_settings."""
        config_data = {
            'models': [],
            'display_settings': {
                'show_token_usage': False,
                'show_reasoning': True,
                'auto_open_changed_files': True,
                'markdown_output': True,
                'file_monitor_include': ['*.py'],
                'file_monitor_exclude': ['*.pyc']
            }
        }
        config = UserConfig(**config_data)
        self.assertFalse(config.display_settings.show_token_usage)
        self.assertTrue(config.display_settings.show_reasoning)
        self.assertTrue(config.display_settings.auto_open_changed_files)
        self.assertTrue(config.display_settings.markdown_output)
        self.assertEqual(config.display_settings.file_monitor_include, ['*.py'])
        self.assertEqual(config.display_settings.file_monitor_exclude, ['*.pyc'])


if __name__ == '__main__':
    unittest.main()