import unittest

from langchain_core.messages import AIMessage, HumanMessage

from llm_workers.utils import format_as_yaml, parse_standard_type, _split_type_parameters, format_tool_args, matches_patterns


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
        self.assertNotIn("Second line", trimmed)
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


class TestFormatToolArgs(unittest.TestCase):
    def test_basic_formatting(self):
        """Test basic argument formatting with simple types."""
        result = format_tool_args(
            {'message': 'Hello', 'mode': 0, 'payload': [0, 0, 1]},
            ['*'],
            max_length=60)
        self.assertEqual("'message': 'Hello', 'mode': 0, 'payload': [0, 0, 1]", result)

    def test_filtering_secrets(self):
        """Test that secret patterns are filtered out."""
        result = format_tool_args(
            {'username': 'alice', 'password': 'secret123'},
            ['*', '!*password*'],
            max_length=60
        )
        self.assertEqual("'username': 'alice'", result)

    def test_multiple_secret_patterns(self):
        """Test filtering with multiple secret patterns (only exclusions = implicit *)."""
        result = format_tool_args(
            {
                'username': 'bob',
                'api_key': 'sk-123',
                'auth_token': 'tok-456',
                'SECRET_VALUE': 'hidden',
                'message': 'Hello World'
            },
            ['!*password*', '!*secret*', '!*SECRET*', '!*_key', '!*token*'],
            max_length=60
        )
        self.assertEqual("'username': 'bob', 'message': 'Hello World'", result)

    def test_truncation(self):
        """Test that long argument strings are truncated."""
        result = format_tool_args(
            {'a': 'very-very-very-very-long-message-here', 'b': 'another-long-value'},
            ['*'],
            max_length=30
        )
        self.assertEqual("'b': 'another-long-value', [...]", result)

    def test_empty_inputs(self):
        """Test with empty input dictionary."""
        result = format_tool_args({}, ['*'], max_length=60)
        self.assertEqual(result, '')

    def test_no_matching_patterns(self):
        """Test when no arguments match the patterns."""
        result = format_tool_args(
            {'foo': 'bar', 'baz': 'qux'},
            ['nonexistent*'],
            max_length=60
        )
        self.assertEqual(result, '')


class TestMatchesPatterns(unittest.TestCase):
    """Tests for matches_patterns function.

    Logic: To match, string must match at least one positive pattern
    AND not match any negative pattern (prefixed with !).
    """

    def test_empty_patterns_returns_false(self):
        """Empty pattern list never matches."""
        self.assertFalse(matches_patterns("any_tool", []))

    def test_only_negative_patterns_matches_by_default(self):
        """Only negative patterns = implicit '*', matches unless excluded."""
        # Not excluded -> matches
        self.assertTrue(matches_patterns("foo", ["!bar"]))
        self.assertTrue(matches_patterns("anything", ["!x", "!y", "!z"]))
        # Excluded -> no match
        self.assertFalse(matches_patterns("foo", ["!foo"]))
        self.assertFalse(matches_patterns("x", ["!x", "!y", "!z"]))

    def test_simple_positive_match(self):
        """Single positive pattern that matches."""
        self.assertTrue(matches_patterns("foo", ["foo"]))
        self.assertTrue(matches_patterns("foo_bar", ["foo*"]))
        self.assertTrue(matches_patterns("bar_foo_baz", ["*foo*"]))

    def test_simple_positive_no_match(self):
        """Single positive pattern that doesn't match."""
        self.assertFalse(matches_patterns("foo", ["bar"]))
        self.assertFalse(matches_patterns("foo", ["bar*"]))
        self.assertFalse(matches_patterns("foo", ["*bar"]))

    def test_multiple_positive_patterns_any_matches(self):
        """Multiple positive patterns - matches if any one matches."""
        self.assertTrue(matches_patterns("foo", ["foo", "bar"]))
        self.assertTrue(matches_patterns("bar", ["foo", "bar"]))
        self.assertTrue(matches_patterns("baz", ["foo", "bar", "baz"]))
        self.assertFalse(matches_patterns("qux", ["foo", "bar", "baz"]))

    def test_positive_with_exclusion_basic(self):
        """Positive match excluded by negative pattern."""
        # Matches gh* but excluded by !gh_write*
        self.assertTrue(matches_patterns("gh_read", ["gh*", "!gh_write*"]))
        self.assertFalse(matches_patterns("gh_write_file", ["gh*", "!gh_write*"]))
        self.assertFalse(matches_patterns("gh_write", ["gh*", "!gh_write*"]))

    def test_positive_with_exclusion_no_initial_match(self):
        """No positive match, exclusion doesn't matter."""
        self.assertFalse(matches_patterns("other_tool", ["gh*", "!gh_write*"]))

    def test_multiple_exclusions(self):
        """Multiple negative patterns all apply."""
        patterns = ["*", "!secret*", "!password*", "!*_key"]
        self.assertTrue(matches_patterns("username", patterns))
        self.assertTrue(matches_patterns("message", patterns))
        self.assertFalse(matches_patterns("secret_value", patterns))
        self.assertFalse(matches_patterns("password", patterns))
        self.assertFalse(matches_patterns("api_key", patterns))

    def test_wildcard_positive_with_exclusions(self):
        """Wildcard * matches all, then exclusions filter."""
        patterns = ["*", "!admin*"]
        self.assertTrue(matches_patterns("user", patterns))
        self.assertTrue(matches_patterns("guest", patterns))
        self.assertFalse(matches_patterns("admin", patterns))
        self.assertFalse(matches_patterns("admin_panel", patterns))

    def test_exact_match_patterns(self):
        """Exact match (no wildcards) in positive patterns."""
        self.assertTrue(matches_patterns("foo", ["foo"]))
        self.assertFalse(matches_patterns("foo_bar", ["foo"]))
        self.assertFalse(matches_patterns("foobar", ["foo"]))

    def test_case_sensitivity(self):
        """Pattern matching is case-sensitive."""
        self.assertTrue(matches_patterns("Foo", ["Foo"]))
        self.assertFalse(matches_patterns("foo", ["Foo"]))
        self.assertFalse(matches_patterns("FOO", ["foo"]))


if __name__ == "__main__":
    unittest.main()