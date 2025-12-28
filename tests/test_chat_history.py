import os
import shutil
import tempfile
import unittest

import yaml
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage,
    AIMessageChunk, ToolMessageChunk
)
from pydantic import ValidationError

from llm_workers.chat_history import ChatHistory, _message_discriminator


class TestChatHistory(unittest.TestCase):
    """Comprehensive tests for ChatHistory module."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file_counter = 0

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def _get_temp_filename(self, name="test"):
        """Generate unique temp filename."""
        self.test_file_counter += 1
        return os.path.join(self.temp_dir, f"{name}_{self.test_file_counter}")

    def _save_and_load_history(self, history: ChatHistory) -> ChatHistory:
        """Helper for round-trip save/load testing."""
        file_path = self._get_temp_filename()
        saved_path = history.save_to_yaml(file_path)
        return ChatHistory.load_from_yaml(saved_path)

    # ===================================================================
    # 1. Discriminator Function Tests
    # ===================================================================

    def test_message_discriminator_human_message(self):
        """Test discriminator returns correct type for HumanMessage."""
        msg = HumanMessage(content="Hello")
        self.assertEqual(_message_discriminator(msg), "human")

    def test_message_discriminator_ai_message(self):
        """Test discriminator returns correct type for AIMessage."""
        msg = AIMessage(content="Hi there")
        self.assertEqual(_message_discriminator(msg), "ai")

    def test_message_discriminator_tool_message(self):
        """Test discriminator returns correct type for ToolMessage."""
        msg = ToolMessage(content="result", tool_call_id="123", name="tool")
        self.assertEqual(_message_discriminator(msg), "tool")

    def test_message_discriminator_ai_message_chunk(self):
        """Test discriminator returns correct type for AIMessageChunk."""
        msg = AIMessageChunk(content="chunk")
        self.assertEqual(_message_discriminator(msg), "AIMessageChunk")

    def test_message_discriminator_tool_message_chunk(self):
        """Test discriminator returns correct type for ToolMessageChunk."""
        msg = ToolMessageChunk(content="chunk", tool_call_id="123", name="tool")
        self.assertEqual(_message_discriminator(msg), "ToolMessageChunk")

    def test_message_discriminator_dict_with_type(self):
        """Test discriminator returns type from dict."""
        msg_dict = {"type": "human", "content": "test"}
        self.assertEqual(_message_discriminator(msg_dict), "human")

    def test_message_discriminator_invalid_object(self):
        """Test discriminator returns None for non-message object."""
        self.assertIsNone(_message_discriminator("string"))
        self.assertIsNone(_message_discriminator(123))
        self.assertIsNone(_message_discriminator(None))

    # ===================================================================
    # 2. Filename Normalization Tests
    # ===================================================================

    def test_normalize_filename_adds_suffix(self):
        """Test normalization adds .chat.yaml suffix."""
        result = ChatHistory._normalize_session_filename("test")
        self.assertEqual(result, "test.chat.yaml")

    def test_normalize_filename_preserves_suffix(self):
        """Test normalization preserves existing .chat.yaml suffix."""
        result = ChatHistory._normalize_session_filename("test.chat.yaml")
        self.assertEqual(result, "test.chat.yaml")

    def test_normalize_filename_with_path(self):
        """Test normalization handles paths correctly."""
        result = ChatHistory._normalize_session_filename("/path/to/test")
        self.assertEqual(result, "/path/to/test.chat.yaml")

    def test_normalize_filename_with_other_extension(self):
        """Test normalization appends suffix to other extensions."""
        result = ChatHistory._normalize_session_filename("test.yaml")
        self.assertEqual(result, "test.yaml.chat.yaml")

    # ===================================================================
    # 3. Basic Serialization Tests
    # ===================================================================

    def test_save_and_load_empty_history(self):
        """Test save/load with empty message list."""
        history = ChatHistory(script_name="test_script", messages=[])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.script_name, "test_script")
        self.assertEqual(len(loaded.messages), 0)

    def test_save_and_load_simple_human_message(self):
        """Test save/load with single HumanMessage."""
        msg = HumanMessage(content="Hello, world!")
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.script_name, "test_script")
        self.assertEqual(len(loaded.messages), 1)
        self.assertIsInstance(loaded.messages[0], HumanMessage)
        self.assertEqual(loaded.messages[0].content, "Hello, world!")

    def test_save_and_load_simple_ai_message(self):
        """Test save/load with simple AIMessage."""
        msg = AIMessage(content="Hi there!")
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(len(loaded.messages), 1)
        self.assertIsInstance(loaded.messages[0], AIMessage)
        self.assertEqual(loaded.messages[0].content, "Hi there!")

    def test_save_and_load_simple_tool_message(self):
        """Test save/load with ToolMessage."""
        msg = ToolMessage(content="result", tool_call_id="123", name="test_tool")
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(len(loaded.messages), 1)
        self.assertIsInstance(loaded.messages[0], ToolMessage)
        self.assertEqual(loaded.messages[0].content, "result")
        self.assertEqual(loaded.messages[0].tool_call_id, "123")
        self.assertEqual(loaded.messages[0].name, "test_tool")

    # ===================================================================
    # 4. AIMessageChunk and ToolMessageChunk Tests
    # ===================================================================

    def test_save_and_load_ai_message_chunk(self):
        """Test save/load with AIMessageChunk preserves chunk type."""
        msg = AIMessageChunk(content="Chunk content")
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(len(loaded.messages), 1)
        # AIMessageChunk is now properly preserved during serialization
        self.assertIsInstance(loaded.messages[0], AIMessageChunk)
        self.assertEqual(loaded.messages[0].content, "Chunk content")

    def test_save_and_load_tool_message_chunk(self):
        """Test save/load with ToolMessageChunk preserves chunk type."""
        msg = ToolMessageChunk(content="chunk result", tool_call_id="456", name="chunk_tool")
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(len(loaded.messages), 1)
        # ToolMessageChunk is now properly preserved during serialization
        self.assertIsInstance(loaded.messages[0], ToolMessageChunk)
        self.assertEqual(loaded.messages[0].content, "chunk result")
        self.assertEqual(loaded.messages[0].tool_call_id, "456")
        self.assertEqual(loaded.messages[0].name, "chunk_tool")

    def test_ai_message_chunk_with_attributes(self):
        """Test AIMessageChunk with additional attributes is preserved."""
        msg = AIMessageChunk(
            content="streaming",
            id="chunk-id-123"
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertIsInstance(loaded.messages[0], AIMessageChunk)
        self.assertEqual(loaded.messages[0].content, "streaming")
        self.assertEqual(loaded.messages[0].id, "chunk-id-123")

    # ===================================================================
    # 5. Complex AIMessage Tests
    # ===================================================================

    def test_save_and_load_ai_message_with_tool_calls(self):
        """Test AIMessage with tool_calls array."""
        msg = AIMessage(
            content="I'll use the tool",
            tool_calls=[{
                "name": "test_tool",
                "args": {"key": "value", "number": 42},
                "id": "call-123",
                "type": "tool_call"
            }]
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertEqual(len(loaded_msg.tool_calls), 1)
        self.assertEqual(loaded_msg.tool_calls[0]["name"], "test_tool")
        self.assertEqual(loaded_msg.tool_calls[0]["args"]["key"], "value")
        self.assertEqual(loaded_msg.tool_calls[0]["args"]["number"], 42)
        self.assertEqual(loaded_msg.tool_calls[0]["id"], "call-123")

    def test_save_and_load_ai_message_with_usage_metadata(self):
        """Test AIMessage with usage_metadata."""
        msg = AIMessage(
            content="Response",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "input_token_details": {"cache_read": 10},
                "output_token_details": {"reasoning": 25}
            }
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertEqual(loaded_msg.usage_metadata["input_tokens"], 100)
        self.assertEqual(loaded_msg.usage_metadata["output_tokens"], 50)
        self.assertEqual(loaded_msg.usage_metadata["total_tokens"], 150)
        self.assertEqual(loaded_msg.usage_metadata["input_token_details"]["cache_read"], 10)
        self.assertEqual(loaded_msg.usage_metadata["output_token_details"]["reasoning"], 25)

    def test_save_and_load_ai_message_with_response_metadata(self):
        """Test AIMessage with response_metadata."""
        msg = AIMessage(
            content="Response",
            response_metadata={
                "safety_ratings": [],
                "model_provider": "test_provider",
                "finish_reason": "STOP",
                "model_name": "test-model-v1"
            }
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertEqual(loaded_msg.response_metadata["model_provider"], "test_provider")
        self.assertEqual(loaded_msg.response_metadata["finish_reason"], "STOP")
        self.assertEqual(loaded_msg.response_metadata["model_name"], "test-model-v1")

    def test_save_and_load_ai_message_with_additional_kwargs(self):
        """Test AIMessage with additional_kwargs."""
        msg = AIMessage(
            content="Response",
            additional_kwargs={
                "function_call": {
                    "name": "my_function",
                    "arguments": '{"arg": "value"}'
                },
                "custom_field": "custom_value"
            }
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertEqual(loaded_msg.additional_kwargs["function_call"]["name"], "my_function")
        self.assertEqual(loaded_msg.additional_kwargs["custom_field"], "custom_value")

    def test_save_and_load_ai_message_with_tool_call_chunks(self):
        """Test AIMessage with tool_call_chunks."""
        msg = AIMessage(
            content=[],
            tool_call_chunks=[{
                "name": "test_tool",
                "args": '{"key": "value"}',
                "id": "chunk-123",
                "type": "tool_call_chunk"
            }]
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertEqual(len(loaded_msg.tool_call_chunks), 1)
        self.assertEqual(loaded_msg.tool_call_chunks[0]["name"], "test_tool")
        self.assertEqual(loaded_msg.tool_call_chunks[0]["id"], "chunk-123")

    def test_save_and_load_ai_message_with_id(self):
        """Test AIMessage with custom id field."""
        msg = AIMessage(
            content="Response",
            id="lc_run--custom-id-12345"
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.messages[0].id, "lc_run--custom-id-12345")

    def test_save_and_load_ai_message_with_chunk_position(self):
        """Test AIMessage with chunk_position field."""
        msg = AIMessage(
            content="Final chunk",
            chunk_position="last"
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        # Note: chunk_position might not be preserved as it's not a standard field
        # This test verifies the message loads without error
        self.assertEqual(loaded.messages[0].content, "Final chunk")

    def test_save_and_load_ai_message_content_as_list(self):
        """Test AIMessage with content as list of dicts."""
        msg = AIMessage(
            content=[{
                "type": "text",
                "text": "The answer is 42.",
                "index": 0
            }]
        )
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_content = loaded.messages[0].content
        self.assertIsInstance(loaded_content, list)
        self.assertEqual(len(loaded_content), 1)
        self.assertEqual(loaded_content[0]["type"], "text")
        self.assertEqual(loaded_content[0]["text"], "The answer is 42.")

    def test_save_and_load_ai_message_empty_content(self):
        """Test AIMessage with empty list content."""
        msg = AIMessage(content=[])
        history = ChatHistory(script_name="test_script", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_content = loaded.messages[0].content
        self.assertIsInstance(loaded_content, list)
        self.assertEqual(len(loaded_content), 0)

    # ===================================================================
    # 6. Multiple Messages Tests
    # ===================================================================

    def test_save_and_load_mixed_message_types(self):
        """Test save/load with multiple message types in sequence."""
        messages = [
            HumanMessage(content="Calculate something"),
            AIMessage(
                content="I'll use a tool",
                tool_calls=[{
                    "name": "calculator",
                    "args": {"expression": "2+2"},
                    "id": "call-1",
                    "type": "tool_call"
                }]
            ),
            ToolMessage(content="4", tool_call_id="call-1", name="calculator"),
            AIMessage(content="The answer is 4")
        ]
        history = ChatHistory(script_name="test_script", messages=messages)
        loaded = self._save_and_load_history(history)

        self.assertEqual(len(loaded.messages), 4)
        self.assertIsInstance(loaded.messages[0], HumanMessage)
        self.assertIsInstance(loaded.messages[1], AIMessage)
        self.assertIsInstance(loaded.messages[2], ToolMessage)
        self.assertIsInstance(loaded.messages[3], AIMessage)

        # Verify content and order preserved
        self.assertEqual(loaded.messages[0].content, "Calculate something")
        self.assertEqual(loaded.messages[2].content, "4")
        self.assertEqual(loaded.messages[3].content, "The answer is 4")

    def test_save_and_load_realistic_conversation(self):
        """Test save/load with realistic conversation structure."""
        messages = [
            HumanMessage(content="Calculate factorial of 10"),
            AIMessage(
                content=[],
                additional_kwargs={
                    "function_call": {
                        "name": "show_plan",
                        "arguments": '{"prompt": "I will calculate..."}'
                    }
                },
                response_metadata={
                    "safety_ratings": [],
                    "model_provider": "test_provider",
                    "finish_reason": "STOP",
                    "model_name": "test-model"
                },
                id="lc_run--test-id-1",
                tool_calls=[{
                    "name": "show_plan",
                    "args": {"prompt": "I will calculate..."},
                    "id": "call-1",
                    "type": "tool_call"
                }],
                usage_metadata={
                    "input_tokens": 311,
                    "output_tokens": 176,
                    "total_tokens": 487
                }
            ),
            ToolMessage(
                content='{"approval_token": "abc123"}',
                tool_call_id="call-1",
                name="show_plan"
            ),
            AIMessage(
                content=[{
                    "type": "text",
                    "text": "The factorial of 10 is 3,628,800.",
                    "index": 0
                }],
                response_metadata={
                    "finish_reason": "STOP",
                    "model_provider": "test_provider"
                },
                usage_metadata={
                    "input_tokens": 745,
                    "output_tokens": 18,
                    "total_tokens": 763
                }
            )
        ]
        history = ChatHistory(script_name="factorial_script", messages=messages)
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.script_name, "factorial_script")
        self.assertEqual(len(loaded.messages), 4)

        # Verify first AI message complexity
        ai_msg = loaded.messages[1]
        self.assertEqual(ai_msg.id, "lc_run--test-id-1")
        self.assertEqual(ai_msg.tool_calls[0]["name"], "show_plan")
        self.assertEqual(ai_msg.usage_metadata["total_tokens"], 487)

        # Verify tool message
        tool_msg = loaded.messages[2]
        self.assertIn("approval_token", tool_msg.content)

        # Verify final AI message with list content
        final_msg = loaded.messages[3]
        self.assertIsInstance(final_msg.content, list)
        self.assertEqual(final_msg.content[0]["text"], "The factorial of 10 is 3,628,800.")

    # ===================================================================
    # 7. Edge Cases and Error Handling Tests
    # ===================================================================

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ChatHistory.load_from_yaml("/nonexistent/path/test.chat.yaml")

    def test_load_from_invalid_yaml(self):
        """Test loading from invalid YAML raises error."""
        file_path = self._get_temp_filename() + ".chat.yaml"
        with open(file_path, 'w') as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(yaml.YAMLError):
            ChatHistory.load_from_yaml(file_path)

    def test_load_from_missing_script_name(self):
        """Test loading YAML missing script_name raises ValidationError."""
        file_path = self._get_temp_filename() + ".chat.yaml"
        with open(file_path, 'w') as f:
            yaml.dump({"messages": []}, f)

        with self.assertRaises(ValidationError):
            ChatHistory.load_from_yaml(file_path)

    def test_load_from_invalid_message_type(self):
        """Test loading with invalid message type raises ValidationError."""
        file_path = self._get_temp_filename() + ".chat.yaml"
        with open(file_path, 'w') as f:
            yaml.dump({
                "script_name": "test",
                "messages": [{"type": "invalid_type", "content": "test"}]
            }, f)

        with self.assertRaises(ValidationError):
            ChatHistory.load_from_yaml(file_path)

    def test_save_returns_normalized_filename(self):
        """Test save_to_yaml returns normalized filename."""
        history = ChatHistory(script_name="test", messages=[])
        file_path = self._get_temp_filename("myfile")

        result_path = history.save_to_yaml(file_path)

        self.assertTrue(result_path.endswith(".chat.yaml"))
        self.assertTrue(os.path.exists(result_path))

    def test_message_with_unicode_content(self):
        """Test messages with unicode characters."""
        msg = HumanMessage(content="Hello 世界 🌍 Привет")
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.messages[0].content, "Hello 世界 🌍 Привет")

    def test_message_with_multiline_content(self):
        """Test messages with multiline strings."""
        multiline = """Line 1
Line 2
Line 3"""
        msg = HumanMessage(content=multiline)
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.messages[0].content, multiline)

    def test_message_with_special_yaml_characters(self):
        """Test messages with YAML special characters."""
        special_content = "Key: value - item | multiline > text & symbols"
        msg = HumanMessage(content=special_content)
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        self.assertEqual(loaded.messages[0].content, special_content)

    def test_multiple_save_load_cycles(self):
        """Test multiple round-trips preserve data."""
        msg = AIMessage(
            content="Test",
            tool_calls=[{"name": "tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}]
        )
        history = ChatHistory(script_name="test", messages=[msg])

        # First cycle
        loaded1 = self._save_and_load_history(history)
        # Second cycle
        loaded2 = self._save_and_load_history(loaded1)
        # Third cycle
        loaded3 = self._save_and_load_history(loaded2)

        self.assertEqual(loaded3.messages[0].content, "Test")
        self.assertEqual(loaded3.messages[0].tool_calls[0]["name"], "tool")

    # ===================================================================
    # 8. Round-Trip Equality Tests
    # ===================================================================

    def test_human_message_equality_after_roundtrip(self):
        """Test HumanMessage fields preserved after round-trip."""
        msg = HumanMessage(content="Test message", id="human-123")
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertIsInstance(loaded_msg, HumanMessage)
        self.assertEqual(loaded_msg.content, msg.content)
        self.assertEqual(loaded_msg.id, msg.id)

    def test_ai_message_equality_after_roundtrip(self):
        """Test complex AIMessage fields preserved after round-trip."""
        msg = AIMessage(
            content=[{"type": "text", "text": "Response"}],
            id="ai-456",
            tool_calls=[{
                "name": "tool1",
                "args": {"param": "value"},
                "id": "call-1",
                "type": "tool_call"
            }],
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 25,
                "total_tokens": 75
            },
            response_metadata={
                "model_provider": "test",
                "finish_reason": "STOP"
            },
            additional_kwargs={
                "custom": "field"
            }
        )
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertIsInstance(loaded_msg, AIMessage)
        self.assertEqual(loaded_msg.content, msg.content)
        self.assertEqual(loaded_msg.id, msg.id)
        self.assertEqual(loaded_msg.tool_calls, msg.tool_calls)
        self.assertEqual(loaded_msg.usage_metadata, msg.usage_metadata)
        self.assertEqual(loaded_msg.response_metadata, msg.response_metadata)
        self.assertEqual(loaded_msg.additional_kwargs, msg.additional_kwargs)

    def test_tool_message_equality_after_roundtrip(self):
        """Test ToolMessage fields preserved after round-trip."""
        msg = ToolMessage(
            content='{"result": "success"}',
            tool_call_id="call-789",
            name="my_tool",
            id="tool-msg-123"
        )
        history = ChatHistory(script_name="test", messages=[msg])
        loaded = self._save_and_load_history(history)

        loaded_msg = loaded.messages[0]
        self.assertIsInstance(loaded_msg, ToolMessage)
        self.assertEqual(loaded_msg.content, msg.content)
        self.assertEqual(loaded_msg.tool_call_id, msg.tool_call_id)
        self.assertEqual(loaded_msg.name, msg.name)
        self.assertEqual(loaded_msg.id, msg.id)


if __name__ == "__main__":
    unittest.main()
