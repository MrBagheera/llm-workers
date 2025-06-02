import unittest
from unittest.mock import patch
from llm_workers.tools.misc import UserInputTool


class TestUserInputTool(unittest.TestCase):

    def test_user_input_tool_schema(self):
        """Test that the UserInputTool has the correct schema."""
        tool = UserInputTool()
        self.assertEqual(tool.name, "user_input")
        self.assertIn("prompt", tool.args_schema.model_fields)
        self.assertTrue(tool.args_schema.model_fields["prompt"].is_required())

    def test_user_input_tool_needs_confirmation(self):
        """Test that UserInputTool doesn't require confirmation."""
        tool = UserInputTool()
        self.assertFalse(tool.needs_confirmation({"prompt": "Test prompt"}))

    def test_user_input_tool_ui_hint(self):
        """Test the UI hint message."""
        tool = UserInputTool()
        hint = tool.get_ui_hint({"prompt": "Test prompt"})
        self.assertEqual(hint, "Requesting user input")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_user_input_tool_single_line(self, mock_print, mock_input):
        """Test UserInputTool with single line input."""
        mock_input.side_effect = ["Hello world", ""]
        
        tool = UserInputTool()
        result = tool._run("Please enter some text:")
        
        self.assertEqual(result, "Hello world")
        mock_print.assert_any_call("Please enter some text:")
        mock_print.assert_any_call("(Enter your input below, use an empty line to finish)")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_user_input_tool_multi_line(self, mock_print, mock_input):
        """Test UserInputTool with multi-line input."""
        mock_input.side_effect = ["Line 1", "Line 2", "Line 3", ""]
        
        tool = UserInputTool()
        result = tool._run("Please enter multiple lines:")
        
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_user_input_tool_empty_input(self, mock_print, mock_input):
        """Test UserInputTool with immediate empty line."""
        mock_input.side_effect = [""]
        
        tool = UserInputTool()
        result = tool._run("Enter something:")
        
        self.assertEqual(result, "")

    @patch('builtins.input')
    @patch('builtins.print')
    def test_user_input_tool_eof_error(self, mock_print, mock_input):
        """Test UserInputTool handles EOF gracefully."""
        mock_input.side_effect = ["Line 1", EOFError()]
        
        tool = UserInputTool()
        result = tool._run("Enter text:")
        
        self.assertEqual(result, "Line 1")