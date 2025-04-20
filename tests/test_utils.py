import unittest

from langchain_core.messages import AIMessage, HumanMessage

from llm_workers.utils import format_as_yaml


class TestFormatMessageAsYaml(unittest.TestCase):
    def test_format_human_message(self):
        # Test with a simple human message
        message = HumanMessage(content="Hello, world!")
        result = format_as_yaml(message, trim=True)

        # The result should contain the content and type
        self.assertIn("content: Hello, world!", result)
        self.assertIn("type: human", result)

    def test_format_ai_message(self):
        # Test with an AI message
        message = AIMessage(content="I'm an AI assistant")
        result = format_as_yaml(message, trim=True)

        # The result should contain the content and type
        self.assertIn("content: I'm an AI assistant", result)
        self.assertIn("type: ai", result)

    def test_trim_long_content(self):
        # Test with a long message that needs trimming
        end_marker = "<end_marker>"
        long_content = "This is a very long message " * 20 + end_marker
        message = HumanMessage(content=long_content)

        # Without trimming
        untrimmed = format_as_yaml(message, trim=False)
        self.assertIn(end_marker, untrimmed)

        # With trimming
        trimmed = format_as_yaml(message, trim=True)
        self.assertNotIn(end_marker, trimmed)
        self.assertIn("This is a very long message", trimmed)
        self.assertIn("...", trimmed)

    def test_trim_multiline_content(self):
        # Test with multiline content
        multiline = "First line\nSecond line\nThird line\nFourth line"
        message = AIMessage(content=multiline)

        # Without trimming
        untrimmed = format_as_yaml(message, trim=False)
        self.assertIn("First line", untrimmed)
        self.assertIn("Second line", untrimmed)
        self.assertIn("Fourth line", untrimmed)

        # With trimming
        trimmed = format_as_yaml(message, trim=True)
        self.assertIn("First line", trimmed)
        self.assertIn("Second line", trimmed)
        self.assertNotIn("Fourth line", trimmed)

    def test_nested_content_structure(self):
        # Test with a message containing nested data
        message = AIMessage(
            content="Response with data",
            additional_kwargs={
                "citations": ["source1", "source2"],
                "metadata": {"confidence": 0.95, "model": "gpt-4"}
            }
        )

        result = format_as_yaml(message, trim=True)

        # Check that nested elements are present
        self.assertIn("additional_kwargs:", result)
        self.assertIn("citations:", result)
        self.assertIn("- source1", result)
        self.assertIn("metadata:", result)
        self.assertIn("confidence:", result)

    def test_nested_content_trimming(self):
        # Test trimming with deeply nested structure
        message = AIMessage(
            content="Main content",
            additional_kwargs={
                "details": {
                    "long_text": "This is a very long nested text " * 10,
                    "multiline": "First\nSecond\nThird\nFourth",
                }
            }
        )

        # With trimming
        trimmed = format_as_yaml(message, trim=True)
        self.assertIn("This is a very long nested text", trimmed)
        self.assertIn("...", trimmed)
        self.assertIn("First", trimmed)
        self.assertNotIn("Fourth", trimmed)


if __name__ == "__main__":
    unittest.main()