import os
import unittest

from llm_workers.config import MCPServerStdio, MCPServerHttp
from llm_workers.expressions import JsonExpression


class TestMCPExpressions(unittest.TestCase):

    def test_env_var_dot_notation(self):
        """Test ${env.VAR} syntax (simpleeval's sweetener feature)"""
        os.environ['TEST_VAR_DOT'] = 'test_value_dot'
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=['${env.TEST_VAR_DOT}'],
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.args.evaluate(eval_context)
        self.assertEqual(result, ['test_value_dot'])
        # Cleanup
        del os.environ['TEST_VAR_DOT']

    def test_env_var_bracket_notation(self):
        """Test ${env['VAR']} syntax"""
        os.environ['TEST_VAR_BRACKET'] = 'test_value_bracket'
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=["${env['TEST_VAR_BRACKET']}"],
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.args.evaluate(eval_context)
        self.assertEqual(result, ['test_value_bracket'])
        # Cleanup
        del os.environ['TEST_VAR_BRACKET']

    def test_mixed_expressions_in_args(self):
        """Test complex args with multiple expressions"""
        os.environ['TEST_HOME'] = '/home/user'
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=['-e', 'HOME=${env.TEST_HOME}', 'static-arg'],
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.args.evaluate(eval_context)
        self.assertEqual(result, ['-e', 'HOME=/home/user', 'static-arg'])
        # Cleanup
        del os.environ['TEST_HOME']

    def test_env_dict_substitution(self):
        """Test environment variable substitution in env dict"""
        os.environ['TEST_SECRET_KEY'] = 'secret123'
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=[],
            env={"SECRET_KEY": "${env.TEST_SECRET_KEY}", "STATIC": "value"},
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.env.evaluate(eval_context)
        self.assertEqual(result, {"SECRET_KEY": "secret123", "STATIC": "value"})
        # Cleanup
        del os.environ['TEST_SECRET_KEY']

    def test_http_headers_substitution(self):
        """Test environment variable substitution in HTTP headers"""
        os.environ['TEST_AUTH_TOKEN'] = 'bearer_token_123'
        server = MCPServerHttp(
            transport='streamable_http',
            url='https://example.com',
            headers={"Authorization": "Bearer ${env.TEST_AUTH_TOKEN}"},
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.headers.evaluate(eval_context)
        self.assertEqual(result, {"Authorization": "Bearer bearer_token_123"})
        # Cleanup
        del os.environ['TEST_AUTH_TOKEN']

    def test_static_args_no_evaluation_needed(self):
        """Test that static args work without any expressions"""
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=['arg1', 'arg2', 'arg3'],
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.args.evaluate(eval_context)
        self.assertEqual(result, ['arg1', 'arg2', 'arg3'])

    def test_mixed_types_in_args(self):
        """Test that args can contain mixed types"""
        os.environ['TEST_PORT'] = '8080'
        server = MCPServerStdio(
            transport='stdio',
            command='test',
            args=['--port', '${env.TEST_PORT}', '--verbose'],
            auto_import_scope='none'
        )
        eval_context = {"env": dict(os.environ)}
        result = server.args.evaluate(eval_context)
        self.assertEqual(result, ['--port', '8080', '--verbose'])
        # Cleanup
        del os.environ['TEST_PORT']

    def test_expression_type_preservation(self):
        """Test that single expression preserves type (not in env context, but general)"""
        # This tests the general StringExpression behavior
        from llm_workers.expressions import StringExpression

        # Single expression should preserve type
        expr = StringExpression("${10}")
        result = expr.evaluate({})
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)

        # Expression with text should return string
        expr2 = StringExpression("Port: ${10}")
        result2 = expr2.evaluate({})
        self.assertEqual(result2, "Port: 10")
        self.assertIsInstance(result2, str)


if __name__ == '__main__':
    unittest.main()
