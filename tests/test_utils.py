import unittest

from langchain_core.messages import AIMessage, HumanMessage

from llm_workers.utils import format_as_yaml, parse_standard_type, _split_type_parameters


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


class TestTypeParsing(unittest.TestCase):
    def test_split_type_parameters(self):
        self.assertEqual(_split_type_parameters("str"), ["str"])
        self.assertEqual(_split_type_parameters("str, int"), ["str", "int"])
        self.assertEqual(_split_type_parameters("str, dict[str, int]"), ["str", "dict[str, int]"])
        self.assertEqual(_split_type_parameters("dict[str, int], list[str]"), ["dict[str, int]", "list[str]"])
        self.assertEqual(_split_type_parameters("str, dict[str, dict[str, int]]"), ["str", "dict[str, dict[str, int]]"])

    def test_parse_type_basic(self):
        self.assertEqual(parse_standard_type("str"), str)
        self.assertEqual(parse_standard_type("int"), int)
        self.assertEqual(parse_standard_type("float"), float)
        self.assertEqual(parse_standard_type("bool"), bool)
        self.assertEqual(parse_standard_type("dict"), dict)
        self.assertEqual(parse_standard_type("list"), list)

    def test_parse_type_literal(self):
        from typing import Literal
        literal_type = parse_standard_type("literal:red|green|blue")
        self.assertEqual(literal_type.__origin__, Literal)
        self.assertEqual(literal_type.__args__, ('red', 'green', 'blue'))

    def test_parse_type_parametrized_list(self):
        from typing import List, Dict
        self.assertEqual(parse_standard_type("list[str]"), List[str])
        self.assertEqual(parse_standard_type("list[int]"), List[int])
        self.assertEqual(parse_standard_type("list[dict[str, int]]"), List[Dict[str, int]])

    def test_parse_type_parametrized_dict(self):
        from typing import List, Dict
        self.assertEqual(parse_standard_type("dict[str, int]"), Dict[str, int])
        self.assertEqual(parse_standard_type("dict[str, str]"), Dict[str, str])
        self.assertEqual(parse_standard_type("dict[str, list[int]]"), Dict[str, List[int]])
        self.assertEqual(parse_standard_type("dict[str, dict[str, int]]"), Dict[str, Dict[str, int]])

    def test_parse_type_errors(self):
        with self.assertRaises(ValueError):
            parse_standard_type("unknown_type")
        
        with self.assertRaises(ValueError):
            parse_standard_type("dict[str]")  # Missing second parameter
        
        with self.assertRaises(ValueError):
            parse_standard_type("dict[str, int, float]")  # Too many parameters

if __name__ == "__main__":
    unittest.main()