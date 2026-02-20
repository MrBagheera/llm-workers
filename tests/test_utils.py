import json
import logging
import os
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from llm_workers.utils import format_as_yaml, parse_standard_type, _split_type_parameters, matches_patterns, load_yaml, LazyFormatter
from llm_workers.worker_utils import format_tool_args


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
        # Test with a long message that needs trimming (single line > 80 chars)
        end_marker = "<end_marker>"
        long_content = "This is a very long message " * 20 + end_marker
        message = HumanMessage(content=long_content)

        # Without trimming
        untrimmed = format_as_yaml(message, trim=False)
        self.assertIn(end_marker, untrimmed)

        # With trimming: first 80 chars shown, rest reported as trimmed
        trimmed = format_as_yaml(message, trim=True)
        self.assertNotIn(end_marker, trimmed)
        self.assertIn("This is a very long message", trimmed)
        self.assertIn("...", trimmed)
        self.assertIn("characters trimmed", trimmed)

    def test_trim_multiline_content(self):
        # Test with multiline content
        multiline = "First line\nSecond line\nThird line\nFourth line"
        message = AIMessage(content=multiline)

        # Without trimming
        untrimmed = format_as_yaml(message, trim=False)
        self.assertIn("First line", untrimmed)
        self.assertIn("Second line", untrimmed)
        self.assertIn("Fourth line", untrimmed)

        # With trim=True (max 1 visual line): only first line shown
        trimmed = format_as_yaml(message, trim=True)
        self.assertIn("First line", trimmed)
        self.assertNotIn("Second line", trimmed)
        self.assertNotIn("Fourth line", trimmed)
        self.assertIn("lines", trimmed)
        self.assertNotIn("characters trimmed", trimmed)

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


class TestTrimInt(unittest.TestCase):
    """Tests for format_as_yaml with trim=int (N visual lines, soft-wrapped at 80 chars)."""

    def _content(self, result: str) -> str:
        """Extract the 'content:' value from YAML output for inspection."""
        for line in result.splitlines():
            if line.strip().startswith("content:"):
                return result  # multiline literal block — return whole thing
        return result

    def test_trim_int_short_string_no_trim(self):
        # A string shorter than N*80 chars should be returned unchanged
        message = HumanMessage(content="Short message")
        result = format_as_yaml(message, trim=3)
        self.assertIn("Short message", result)
        self.assertNotIn("trimmed", result)

    def test_trim_int_multiline_within_limit(self):
        # Three short lines with trim=3: all fit, nothing trimmed
        content = "Line one\nLine two\nLine three"
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=3)
        self.assertIn("Line one", result)
        self.assertIn("Line two", result)
        self.assertIn("Line three", result)
        self.assertNotIn("trimmed", result)

    def test_trim_int_multiline_exceeds_limit(self):
        # Five short lines with trim=2: lines 3-5 should be trimmed
        content = "Line one\nLine two\nLine three\nLine four\nLine five"
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=2)
        self.assertIn("Line one", result)
        self.assertIn("Line two", result)
        self.assertNotIn("Line three", result)
        self.assertNotIn("Line five", result)
        self.assertIn("lines", result)
        self.assertNotIn("characters trimmed", result)

    def test_trim_int_message_reports_visual_lines(self):
        # Each short line (<= 80 chars) counts as 1 visual line
        # 4 lines, trim=2 → 2 visual lines trimmed (lines 3 and 4)
        content = "A\nB\nC\nD"
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=2)
        self.assertIn("[2 lines trimmed]", result)

    def test_trim_int_long_line_counts_as_multiple_visual_lines(self):
        # A 160-char line = 2 visual lines; trim=1 should trim part of it
        long_line = "x" * 160
        message = HumanMessage(content=long_line)
        result = format_as_yaml(message, trim=1)
        # Single actual line partially trimmed: 0 full lines omitted, 80 chars trimmed
        self.assertIn("...", result)
        self.assertIn("[80 characters trimmed]", result)

    def test_trim_int_partial_last_line_trimmed(self):
        # Single long line: last-line chars trimmed even though no full lines follow
        content = "x" * 100  # trim=1 → first 80 chars shown, 20 chars trimmed
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=1)
        self.assertIn("...", result)
        self.assertIn("[20 characters trimmed]", result)
        self.assertNotIn("x" * 100, result)  # full string not present

    def test_trim_int_char_budget_spanning_lines(self):
        # trim=2 (budget 160 chars): line1=100 chars uses 100, line2=100 chars → only 60 shown
        line1 = "a" * 100
        line2 = "b" * 100
        content = line1 + "\n" + line2
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=2)
        self.assertIn("a" * 80, result)   # at least the budget chars of line1 shown
        self.assertIn("...", result)       # line2 was partially trimmed
        self.assertIn("characters trimmed", result)

    def test_trim_int_false_still_works(self):
        # trim=False: full multiline preserved as literal block
        content = "Line one\nLine two\nLine three"
        message = HumanMessage(content=content)
        result = format_as_yaml(message, trim=False)
        self.assertIn("Line one", result)
        self.assertIn("Line three", result)
        self.assertNotIn("trimmed", result)


class TestLazyFormatterLogger(unittest.TestCase):
    """Tests for LazyFormatter with the logger parameter."""

    def test_no_logger_uses_trim_value(self):
        # Without logger, trim is used as-is
        content = "Line one\nLine two\nLine three"
        lf = LazyFormatter(content, trim=1)
        result = str(lf)
        self.assertIn("Line one", result)
        self.assertNotIn("Line two", result)
        self.assertIn("2 lines trimmed", result)

    def test_logger_at_debug_level_preserves_trim(self):
        # Logger at DEBUG level (not ALL) → trim is NOT overridden
        log = logging.getLogger("test.lazy.debug")
        log.setLevel(logging.DEBUG)
        content = "Line one\nLine two\nLine three"
        lf = LazyFormatter(content, trim=1, logger=log)
        result = str(lf)
        self.assertIn("Line one", result)
        self.assertNotIn("Line two", result)

    def test_logger_at_notset_disables_trim(self):
        # Logger at NOTSET (ALL) level → trim overridden to False (full output)
        log = logging.getLogger("test.lazy.all")
        log.setLevel(logging.NOTSET)
        # Also set root to NOTSET so getEffectiveLevel() returns 0
        root = logging.getLogger()
        original_root_level = root.level
        root.setLevel(logging.NOTSET)
        try:
            content = "Line one\nLine two\nLine three"
            lf = LazyFormatter(content, trim=1, logger=log)
            self.assertEqual(lf.trim, False)
            result = str(lf)
            self.assertIn("Line one", result)
            self.assertIn("Line two", result)
            self.assertIn("Line three", result)
            self.assertNotIn("trimmed", result)
        finally:
            root.setLevel(original_root_level)

    def test_logger_none_uses_trim_value(self):
        # Explicitly passing logger=None behaves same as no logger
        content = "A\nB\nC"
        lf = LazyFormatter(content, trim=1, logger=None)
        self.assertEqual(lf.trim, 1)


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


class TestLoadYaml(unittest.TestCase):
    """Tests for load_yaml function and SmartLoader with !include and !require tags."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_basic_yaml_loading(self):
        """Test basic YAML file loading."""
        yaml_path = os.path.join(self.temp_dir, "basic.yaml")
        with open(yaml_path, 'w') as f:
            f.write("name: test\nvalue: 42\nlist:\n  - item1\n  - item2\n")

        result = load_yaml(yaml_path)
        self.assertEqual(result['name'], 'test')
        self.assertEqual(result['value'], 42)
        self.assertEqual(result['list'], ['item1', 'item2'])

    def test_include_yaml_file(self):
        """Test !include with a YAML file."""
        # Create included file
        included_path = os.path.join(self.temp_dir, "included.yaml")
        with open(included_path, 'w') as f:
            f.write("included_key: included_value\n")

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("main_key: main_value\nincluded: !include included.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['main_key'], 'main_value')
        self.assertEqual(result['included']['included_key'], 'included_value')

    def test_include_json_file(self):
        """Test !include with a JSON file."""
        # Create JSON file
        json_path = os.path.join(self.temp_dir, "data.json")
        with open(json_path, 'w') as f:
            json.dump({"json_key": "json_value", "number": 123}, f)

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !include data.json\n")

        result = load_yaml(main_path)
        self.assertEqual(result['data']['json_key'], 'json_value')
        self.assertEqual(result['data']['number'], 123)

    def test_include_text_file(self):
        """Test !include with a plain text file."""
        # Create text file
        text_path = os.path.join(self.temp_dir, "content.txt")
        with open(text_path, 'w') as f:
            f.write("This is plain text content.\nWith multiple lines.")

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("text: !include content.txt\n")

        result = load_yaml(main_path)
        self.assertEqual(result['text'], "This is plain text content.\nWith multiple lines.")

    def test_include_missing_file_returns_empty_string(self):
        """Test !include with a missing file returns empty string."""
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("content: !include missing.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['content'], "")

    def test_require_yaml_file(self):
        """Test !require with an existing YAML file."""
        # Create required file
        required_path = os.path.join(self.temp_dir, "required.yaml")
        with open(required_path, 'w') as f:
            f.write("required_key: required_value\n")

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !require required.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['data']['required_key'], 'required_value')

    def test_require_missing_file_raises_error(self):
        """Test !require with a missing file raises FileNotFoundError."""
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !require missing.yaml\n")

        with self.assertRaises(FileNotFoundError):
            load_yaml(main_path)

    def test_nested_includes(self):
        """Test nested !include (YAML file including another YAML file)."""
        # Create nested file
        nested_path = os.path.join(self.temp_dir, "nested.yaml")
        with open(nested_path, 'w') as f:
            f.write("nested_value: deep\n")

        # Create middle file that includes nested
        middle_path = os.path.join(self.temp_dir, "middle.yaml")
        with open(middle_path, 'w') as f:
            f.write("middle_value: medium\nnested: !include nested.yaml\n")

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("main_value: top\nmiddle: !include middle.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['main_value'], 'top')
        self.assertEqual(result['middle']['middle_value'], 'medium')
        self.assertEqual(result['middle']['nested']['nested_value'], 'deep')

    def test_path_relative_to_yaml_file(self):
        """Test that paths are resolved relative to the YAML file directory."""
        # Create a subdirectory
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir)

        # Create included file in subdirectory
        included_path = os.path.join(sub_dir, "included.yaml")
        with open(included_path, 'w') as f:
            f.write("sub_key: sub_value\n")

        # Create main file also in subdirectory
        main_path = os.path.join(sub_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !include included.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['data']['sub_key'], 'sub_value')

    def test_path_relative_to_current_directory(self):
        """Test paths with ./ prefix are resolved relative to current directory."""
        # Create file in temp directory
        included_path = os.path.join(self.temp_dir, "included.yaml")
        with open(included_path, 'w') as f:
            f.write("cwd_key: cwd_value\n")

        # Create subdirectory and main file in it
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir)
        main_path = os.path.join(sub_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !include ./included.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['data']['cwd_key'], 'cwd_value')

    def test_parent_directory_escape_blocked(self):
        """Test that .. in paths is blocked for security."""
        # Create file in parent directory
        parent_file = os.path.join(self.temp_dir, "parent.yaml")
        with open(parent_file, 'w') as f:
            f.write("parent_key: parent_value\n")

        # Create subdirectory and main file
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir)
        main_path = os.path.join(sub_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data: !include ../parent.yaml\n")

        # Should raise ValueError due to security check
        with self.assertRaises(ValueError) as context:
            load_yaml(main_path)
        self.assertIn("cannot escape current directory", str(context.exception))

    def test_multiple_includes(self):
        """Test multiple !include tags in one file."""
        # Create multiple included files
        file1_path = os.path.join(self.temp_dir, "file1.yaml")
        with open(file1_path, 'w') as f:
            f.write("key1: value1\n")

        file2_path = os.path.join(self.temp_dir, "file2.json")
        with open(file2_path, 'w') as f:
            json.dump({"key2": "value2"}, f)

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        with open(file3_path, 'w') as f:
            f.write("plain text")

        # Create main file
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("data1: !include file1.yaml\ndata2: !include file2.json\ndata3: !include file3.txt\n")

        result = load_yaml(main_path)
        self.assertEqual(result['data1']['key1'], 'value1')
        self.assertEqual(result['data2']['key2'], 'value2')
        self.assertEqual(result['data3'], 'plain text')

    def test_include_and_require_mixed(self):
        """Test mixing !include and !require in one file."""
        # Create files
        existing_path = os.path.join(self.temp_dir, "existing.yaml")
        with open(existing_path, 'w') as f:
            f.write("exists: true\n")

        # Main file with both tags
        main_path = os.path.join(self.temp_dir, "main.yaml")
        with open(main_path, 'w') as f:
            f.write("required: !require existing.yaml\noptional: !include missing.yaml\n")

        result = load_yaml(main_path)
        self.assertEqual(result['required']['exists'], True)
        self.assertEqual(result['optional'], "")

    def test_load_yaml_with_path_object(self):
        """Test load_yaml with Path object instead of string."""
        yaml_path = Path(self.temp_dir) / "test.yaml"
        with open(yaml_path, 'w') as f:
            f.write("key: value\n")

        result = load_yaml(yaml_path)
        self.assertEqual(result['key'], 'value')


if __name__ == "__main__":
    unittest.main()