import unittest
import json
from unittest.mock import patch

from llm_workers.tools.misc import (
    RequestApprovalTool,
    ValidateApprovalTool,
    ConsumeApprovalTool,
    _approval_tokens
)


class TestApprovalTools(unittest.TestCase):
    
    def setUp(self):
        """Clear approval tokens before each test."""
        _approval_tokens.clear()
    
    def test_approval_workflow(self):
        """Test complete approval workflow: request -> validate -> consume."""
        # Initialize tools
        request_tool = RequestApprovalTool()
        validate_tool = ValidateApprovalTool()
        consume_tool = ConsumeApprovalTool()
        
        # Test 1: Request approval and get token
        result = request_tool._run("Test approval request")
        token_data = json.loads(result)
        token = token_data["approval_token"]
        
        self.assertIn("approval_token", token_data)
        self.assertEqual(len(_approval_tokens), 1)
        self.assertIn(token, _approval_tokens)
        self.assertEqual("Test approval request", _approval_tokens[token]["data"])
        
        # Test 2: Validate the token
        result = validate_tool._run(token)
        self.assertEqual("Test approval request", result)
        
        # Test 3: Consume the token
        result = consume_tool._run(token)
        self.assertEqual("Approval token consumed", result)
        self.assertNotIn(token, _approval_tokens)
        
        # Test 4: Try to validate consumed token (should fail)
        with self.assertRaises(Exception) as context:
            validate_tool._run(token)
        self.assertIn("Invalid or already consumed", str(context.exception))
        
        # Test 5: Try to consume already consumed token (should fail)
        with self.assertRaises(Exception) as context:
            consume_tool._run(token)
        self.assertIn("Invalid or already consumed", str(context.exception))
    
    def test_invalid_token(self):
        """Test behavior with invalid tokens."""
        validate_tool = ValidateApprovalTool()
        consume_tool = ConsumeApprovalTool()
        
        # Test validation with invalid token
        with self.assertRaises(Exception) as context:
            validate_tool._run("invalid_token")
        self.assertIn("Invalid or already consumed", str(context.exception))
        
        # Test consumption with invalid token
        with self.assertRaises(Exception) as context:
            consume_tool._run("invalid_token")
        self.assertIn("Invalid or already consumed", str(context.exception))
    
    def test_tool_properties(self):
        """Test tool properties and schema."""
        request_tool = RequestApprovalTool()
        validate_tool = ValidateApprovalTool()
        consume_tool = ConsumeApprovalTool()
        
        # Test names
        self.assertEqual(request_tool.name, "request_approval")
        self.assertEqual(validate_tool.name, "validate_approval")
        self.assertEqual(consume_tool.name, "consume_approval")
        
        # Test UI hints are empty
        self.assertEqual(request_tool.get_ui_hint({}), "")
        self.assertEqual(validate_tool.get_ui_hint({}), "")
        self.assertEqual(consume_tool.get_ui_hint({}), "")
        
        # Test that request tool needs confirmation
        self.assertTrue(request_tool.needs_confirmation({"prompt": "test"}))
        self.assertFalse(validate_tool.needs_confirmation({"approval_token": "test"}))
        self.assertFalse(consume_tool.needs_confirmation({"approval_token": "test"}))


if __name__ == '__main__':
    unittest.main()