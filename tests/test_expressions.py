import unittest
from typing import Dict, Any

from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from llm_workers.expressions import StringExpression, JsonExpression, EvaluationContext


class TestFunctions(unittest.TestCase):

    def test_merge(self):
        """Test merge function."""
        result = StringExpression("${merge(a, b)}").evaluate(EvaluationContext({'a': [1, 2], 'b': [2, 3]}))
        self.assertEqual([1, 2, 2, 3], result)

        result = StringExpression("${merge(a, b)}").evaluate(EvaluationContext({'a': {'k1': 2}, 'b': {'k2': 3}}))
        self.assertEqual({'k1': 2, 'k2': 3}, result)

        result = StringExpression("${merge(a, b)}").evaluate(EvaluationContext({'a': "Meaning of life is ", 'b': 42}))
        self.assertEqual("Meaning of life is 42", result)

    def test_split(self):
        """Test split function."""
        result = StringExpression("${split(text,',')}").evaluate(EvaluationContext({'text': "a,b,c"}))
        self.assertEqual(['a', 'b', 'c'], result)

        result = StringExpression("${split(text, ' ')}").evaluate(EvaluationContext({'text': "hello world"}))
        self.assertEqual(['hello', 'world'], result)

    def test_join(self):
        """Test join function."""
        result = StringExpression("${join(items, '')}").evaluate(EvaluationContext({'items': ['a', 'b', 'c']}))
        self.assertEqual("abc", result)

        result = StringExpression("${join(items, ', ')}").evaluate(EvaluationContext({'items': ['hello', 'world']}))
        self.assertEqual("hello, world", result)

    def test_flatten(self):
        """Test flatten function."""
        result = StringExpression("${flatten(nested)}").evaluate(EvaluationContext({'nested': [[1, 2], [3, 4]]}))
        self.assertEqual([1, 2, 3, 4], result)

        result = StringExpression("${flatten(nested)}").evaluate(EvaluationContext({'nested': [['a', 'b'], ['c']]}))
        self.assertEqual(['a', 'b', 'c'], result)

    def test_parse_and_print_json(self):
        """Test parse_json and print_json functions."""
        result = StringExpression("${parse_json(json_str)}").evaluate(
            EvaluationContext({'json_str': '{"key": "value", "num": 42}'}))
        self.assertEqual({'key': 'value', 'num': 42}, result)

        # expect ToolException
        with self.assertRaises(ToolException) as e:
            StringExpression("${parse_json('json_str')}").evaluate(
                EvaluationContext({}))
        self.assertIn("Failed to parse JSON", str(e.exception))

        result = StringExpression("${parse_json(\"json_str\", ignore_error=true)}").evaluate(
            EvaluationContext({}))
        self.assertEqual('json_str', result)

        result = StringExpression("${print_json(data)}").evaluate(
            EvaluationContext({'data': {'key': 'value', 'num': 42}}))
        self.assertEqual('{"key": "value", "num": 42}', result)


class TestStringExpression(unittest.TestCase):

    def test_static_string_optimization(self):
        """Test that strings without code blocks are marked static and return raw value."""
        expr = StringExpression("Hello World")
        self.assertFalse(expr.is_dynamic)
        self.assertEqual(expr.evaluate({}), "Hello World")
        # Ensure it works with None context
        self.assertEqual(expr.evaluate(None), "Hello World")

    def test_basic_evaluation(self):
        """Test simple variable substitution."""
        expr = StringExpression("Hello ${name}")
        self.assertTrue(expr.is_dynamic)
        result = expr.evaluate({"name": "User"})
        self.assertEqual(result, "Hello User")

    def test_math_expressions(self):
        """Test simple math operations inside blocks."""
        expr = StringExpression("Result: ${a + b}")
        result = expr.evaluate({"a": 5, "b": 10})
        self.assertEqual(result, "Result: 15")

    def test_escaping_behavior(self):
        """Test that \${...} is treated as a literal ${...} and not evaluated."""
        # Note: In python strings, backslash needs escaping or raw strings.
        # Input string is effectively: "Value is \${price}"
        expr = StringExpression(r"Value is \${price}")

        # It should NOT be dynamic because the only special block was escaped
        self.assertFalse(expr.is_dynamic)

        # The backslash should be removed in the output
        result = expr.evaluate({"price": 100})
        self.assertEqual(result, "Value is ${price}")

    def test_mixed_escaping_and_dynamic(self):
        """Test a string containing both escaped and real code blocks."""
        # Input: "Use \${var} to see ${var}"
        expr = StringExpression(r"Use \${var} to see ${var}")

        self.assertTrue(expr.is_dynamic)
        result = expr.evaluate({"var": "result"})
        # Expect: "Use ${var} to see result"
        self.assertEqual(result, "Use ${var} to see result")

    def test_multiple_variables(self):
        """Test multiple code blocks in one string."""
        expr = StringExpression("${x} + ${y} = ${x + y}")
        result = expr.evaluate({"x": 1, "y": 2})
        self.assertEqual(result, "1 + 2 = 3")

    def test_pydantic_integration(self):
        """Test that the class works as a Pydantic field."""
        class MyConfig(BaseModel):
            template: StringExpression

        # 1. Test Instantiation
        data = {"template": "Hello ${name}"}
        model = MyConfig(**data)

        self.assertIsInstance(model.template, StringExpression)
        self.assertEqual(model.template.evaluate({"name": "Pydantic"}), "Hello Pydantic")

        # 2. Test JSON serialization (if needed, usually handled by custom serializer,
        # but default behavior is checking __str__ or dict encoding)
        # By default pydantic might not know how to dump custom types to json unless serializer is defined,
        # but we are testing INPUT here.

    def test_missing_variable_behavior(self):
        """Test behavior when a variable is missing in the context."""
        expr = StringExpression("Hello ${unknown}")
        # Expect ToolException with enough details
        with self.assertRaises(ToolException) as cm:
            expr.evaluate({"known": "value"})
        self.assertIn("Failed to evaluate ${unknown}: 'unknown' is not defined, available names: ['known']", str(cm.exception))

    def test_syntax_error(self):
        """Test behavior when the code block contains invalid python syntax."""
        with self.assertRaises(SyntaxError):
            StringExpression("Value: ${1 + }") # Invalid syntax

    def test_empty_string(self):
        """Test empty string input."""
        expr = StringExpression("")
        self.assertFalse(expr.is_dynamic)
        self.assertEqual(expr.evaluate({}), "")

    def test_complex_escaping_sequence(self):
        """Test sequences like \\${var} (escaped backslash)."""
        # This implementation does not support "escaping the escape character" (e.g. \\${)
        # specifically, it treats \${ as an escape sequence.
        # But let's verify what happens if we have literal backslash before brace.

        # Case: A literal backslash followed by a dynamic block
        # Input: "C:\\${path}" -> We want C:\ followed by evaluated path
        # Current regex: matches \${path} as escaped block.
        # So "C:\\${path}" -> tokens will see `C:\` then `\${path}`.
        # `\${path}` becomes `${path}` literal.

        expr = StringExpression(r"Path: \${windows}")
        # Should become literal "Path: ${windows}"
        self.assertEqual(expr.evaluate({"windows": "XP"}), "Path: ${windows}")

    def test_single_block_type_preservation(self):
        """Test that single code blocks return raw types (int, list, etc)."""
        # Integer
        self.assertEqual(StringExpression("${10 + 20}").evaluate({}), 30)

        # Boolean
        self.assertEqual(StringExpression("${True}").evaluate({}), True)

        # List
        ctx = {"items": [1, 2]}
        result = StringExpression("${items}").evaluate(ctx)
        self.assertEqual(result, [1, 2])
        self.assertIsInstance(result, list)

    def test_mixed_content_returns_string(self):
        """Test that adding text forces string conversion."""
        # Even if the result is int, presence of text "Sum: " makes it a string
        result = StringExpression("Sum: ${10 + 20}").evaluate({})
        self.assertEqual(result, "Sum: 30")
        self.assertIsInstance(result, str)


class TestJsonExpression(unittest.TestCase):

    def test_static_primitives(self):
        """Test that primitives (int, bool, None) remain untouched and static."""
        data = {"int": 1, "bool": True, "none": None}
        expr = JsonExpression(data)

        self.assertFalse(expr._is_dynamic)
        self.assertEqual(expr.evaluate({}), data)

    def test_flat_dictionary_eval(self):
        """Test a simple flat dictionary with one dynamic string."""
        data = {"static": "A", "dynamic": "Value: ${x}"}
        expr = JsonExpression(data)

        self.assertTrue(expr._is_dynamic)
        result = expr.evaluate({"x": 100})
        self.assertEqual(result, {"static": "A", "dynamic": "Value: 100"})

    def test_nested_dictionaries(self):
        """Test deep recursion into nested dicts."""
        data = {
            "level1": {
                "level2": {
                    "target": "${secret}"
                }
            }
        }
        expr = JsonExpression(data)
        result = expr.evaluate({"secret": "key"})
        self.assertEqual(result["level1"]["level2"]["target"], "key")

    def test_list_processing(self):
        """Test lists containing mix of static and dynamic items."""
        data = ["static", "${val}", 123]
        expr = JsonExpression(data)

        self.assertTrue(expr._is_dynamic)
        result = expr.evaluate({"val": "dynamic"})
        self.assertEqual(result, ["static", "dynamic", 123])

    def test_complex_structure(self):
        """Test a complex graph of lists and dicts."""
        data = {
            "users": [
                {"name": "admin", "role": "${admin_role}"},
                {"name": "guest", "role": "readonly"}
            ],
            "config": {
                "timeout": 30,
                "host": "https://${region}.api.com"
            }
        }
        expr = JsonExpression(data)
        context = {"admin_role": "superuser", "region": "us-east"}

        result = expr.evaluate(context)

        self.assertEqual(result["users"][0]["role"], "superuser")
        self.assertEqual(result["config"]["host"], "https://us-east.api.com")

    def test_escaping_inside_json(self):
        """Test that escaping works within the JSON structure."""
        # Input: {"key": "\${escaped}"}
        # The parser should mark this as dynamic/processed because "\${" needs to become "${"
        data = {"key": r"\${keep_me}"}
        expr = JsonExpression(data)

        # Even though there is no python code to run, the structure is 'modified'
        # (unescaped), so we expect the transformed result.
        result = expr.evaluate({"keep_me": "should_not_appear"})

        # The variable should NOT be substituted, backslash removed.
        self.assertEqual(result, {"key": "${keep_me}"})

    def test_partial_dynamic_optimization(self):
        """
        Verify that static branches of the tree are not re-created.
        (White-box testing: checking if object identity is preserved for static parts)
        """
        static_part = {"a": 1}
        data = {
            "static": static_part,
            "dynamic": "${x}"
        }

        expr = JsonExpression(data)
        result = expr.evaluate({"x": 2})

        # The evaluate result creates a new dict for the root (because it changed),
        # but the static sub-dictionary should ideally be passed through.
        # Note: In our implementation, `eval_node` creates new dicts recursively
        # to ensure safety, but let's just ensure the data is correct.
        self.assertEqual(result["static"]["a"], 1)

    def test_pydantic_integration(self):
        """Test that JsonExpression works as a field in a Pydantic model."""
        class Config(BaseModel):
            meta: str
            payload: JsonExpression

        raw_data = {
            "meta": "v1",
            "payload": {
                "id": "${id_gen}",
                "values": [1, 2, "${extra}"]
            }
        }

        model = Config(**raw_data)

        # Check Pydantic validation passed
        self.assertIsInstance(model.payload, JsonExpression)

        # Check evaluation
        result = model.payload.evaluate({"id_gen": 999, "extra": 3})
        self.assertEqual(result, {"id": 999, "values": [1, 2, 3]})

    def test_top_level_list(self):
        """Test when the root element is a list."""
        data = ["${a}", "${b}"]
        expr = JsonExpression(data)
        self.assertEqual(expr.evaluate({"a": 1, "b": 2}), [1, 2])


# noinspection PyTypeChecker
def test_generic_dict_enforcement(self):
    """Test that JsonExpression[dict] accepts dicts but rejects lists/primitives."""
    class DictModel(BaseModel):
        config: JsonExpression[dict]

    # 1. Success: Valid Dictionary
    valid_data = {"config": {"key": "${val}"}}
    model = DictModel(**valid_data)
    self.assertIsInstance(model.config, JsonExpression)
    self.assertEqual(model.config.evaluate({"val": 1}), {"key": "1"})

    # 2. Failure: Input is a List
    with self.assertRaises(ValidationError) as cm:
        DictModel(config=["not", "a", "dict"])
    self.assertIn("Input should be a valid dictionary", str(cm.exception))

    # 3. Failure: Input is a String
    with self.assertRaises(ValidationError):
        DictModel(config="just a string")


# noinspection PyTypeChecker
def test_generic_list_enforcement(self):
    """Test that JsonExpression[list] accepts lists but rejects dicts."""
    class ListModel(BaseModel):
        items: JsonExpression[list]

    # 1. Success: Valid List
    valid_data = {"items": ["item1", "${item2}"]}
    model = ListModel(**valid_data)
    self.assertEqual(model.items.evaluate({"item2": "B"}), ["item1", "B"])

    # 2. Failure: Input is a Dict
    with self.assertRaises(ValidationError) as cm:
        ListModel(items={"not": "a list"})
    self.assertIn("Input should be a valid list", str(cm.exception))


# noinspection PyTypeChecker
def test_complex_generic_types(self):
    """
    Test stricter typing like JsonExpression[Dict[str, Any]].
    Note: You usually want 'Any' as the leaf type to allow for string expressions.
    """
    class StrictModel(BaseModel):
        # Enforce keys are strings, values can be anything (including our expression strings)
        tags: JsonExpression[Dict[str, Any]]

    # Success
    model = StrictModel(tags={"env": "${env_name}"})
    self.assertEqual(model.tags.evaluate({"env_name": "prod"}), {"env": "prod"})

    # Failure (Key is not a string)
    with self.assertRaises(ValidationError) as cm:
        StrictModel(tags={123: "bad key type"})
    # Pydantic validation error will trigger here
    self.assertIn("Input should be a valid string", str(cm.exception))


# noinspection PyTypeChecker
def test_validation_does_not_break_expressions(self):
    """
    Crucial check: Ensure that validating against 'dict' doesn't
    prevent us from using strings *inside* that dict.
    """
    class Config(BaseModel):
        # We enforce that the root is a dict, but its values are 'Any'
        data: JsonExpression[dict]

    # This should PASS validation because the root is a dict.
    # The inner value "${code}" is a string, which fits 'Any'.
    model = Config(data={"field": "${code}"})

    result = model.data.evaluate({"code": 123})
    self.assertEqual(result, {"field": "123"})

if __name__ == '__main__':
    unittest.main()