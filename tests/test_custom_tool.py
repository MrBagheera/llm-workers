import unittest

from langchain_core.tools import tool

from llm_workers.tools.custom_tools import TemplateHelper, create_statement_from_model
from llm_workers.config import CustomToolParamsDefinition, CallDefinition, ResultDefinition, \
    MatchDefinition, MatchClauseDefinition


class TestTemplateHelper(unittest.TestCase):

    def test_simple_replacement(self):
        helper = TemplateHelper.from_param_definitions(
            params = [
                CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                CustomToolParamsDefinition(name = "param2", description = "This is the second parameter", type = "integer", default = 42),
            ],
            target_params = {"key1": "{param1} is {param2}", "key2": "{param2}", "key3": "value3"},
        )
        result = helper.render({"param1": "Meaning of life", "param2": 42})
        self.assertDictEqual(result, {"key1": "Meaning of life is 42", "key2": 42, "key3": "value3"})

    def test_nested_replacement(self):
        helper = TemplateHelper.from_param_definitions(
            params = [
                CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                CustomToolParamsDefinition(name = "param2", description = "This is the second parameter", type = "integer", default = 42),
            ],
            target_params = {"key1": ["{param1} is {param2}", "value"], "key2": {"k2": "{param2}", "k3": "value3"}},
        )
        result = helper.render({"param1": "Meaning of life", "param2": 42})
        self.assertDictEqual(result, {"key1": ["Meaning of life is 42", "value"], "key2": {"k2": 42, "k3": "value3"}})

    def test_missing_replacement_when_building1(self):
            with self.assertRaises(ValueError) as ex:
                TemplateHelper.from_param_definitions(
                    params = [
                        CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                    ],
                    target_params = {"key1": ["{param1} is {param2}", "value"], "key2": {"k2": "{param2}", "k3": "value3"}},
                )
            self.assertEqual(str(ex.exception), "Unknown reference {param2} for key key1.0, available params: ['param1']")

    def test_missing_replacement_when_building2(self):
        with self.assertRaises(ValueError) as ex:
            TemplateHelper.from_param_definitions(
                params = [
                    CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                ],
                target_params = {"key1": ["{param1} is 42", "value"], "key2": {"k2": "{param2}", "k3": "value3"}},
            )
        self.assertEqual(str(ex.exception), "Unknown reference {param2} for key key2.k2, available params: ['param1']")

    def test_missing_replacement_when_running1(self):
        helper = TemplateHelper.from_param_definitions(
            params = [
                CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                CustomToolParamsDefinition(name = "param2", description = "This is the second parameter", type = "integer", default = 42),
            ],
            target_params = {"key1": ["{param1} is {param2}", "value"], "key2": {"k2": "{param2}", "k3": "value3"}},
        )
        with self.assertRaises(ValueError) as ex:
            helper.render({"param1": "Meaning of life"})
        self.assertEqual(str(ex.exception), "Missing reference for key key1.0: 'param2'")

    def test_missing_replacement_when_running2(self):
        helper = TemplateHelper.from_param_definitions(
            params = [
                CustomToolParamsDefinition(name = "param1", description = "This is the first parameter", type = "string"),
                CustomToolParamsDefinition(name = "param2", description = "This is the second parameter", type = "integer", default = 42),
            ],
            target_params = {"key1": ["{param1} is 42", "value"], "key2": {"k2": "{param2}", "k3": "value3"}},
        )
        with self.assertRaises(ValueError) as ex:
            helper.render({"param1": "Meaning of life"})
        self.assertEqual(str(ex.exception), "Missing reference {param2} for key key2.k2")


@tool
def test_tool_logic(param1: int, param2: int) -> int:
    """Sum two parameters"""
    return param1 + param2

def no_tool_lookup(tool_ref, config):
    raise ValueError(f"Unexpected tool lookup: {tool_ref}")

class TestStatements(unittest.TestCase):

    def test_return_string(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = ResultDefinition(result="{param1} is 42"),
            tool_factory=no_tool_lookup
        )
        self.assertEqual("Meaning of life is 42", statement.invoke({"param1": "Meaning of life"}))

    def test_return_json(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = ResultDefinition(result={"inner": "{param1} is 42"}),
            tool_factory=no_tool_lookup
        )
        self.assertEqual({"inner": "Meaning of life is 42"}, statement.invoke({"param1": "Meaning of life"}))

    def test_simple_call(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = CallDefinition(call = "some_function", params = {"param1": "{param1}", "param2": 29}),
            tool_factory= lambda tool_ref, config: test_tool_logic
        )
        assert 42 == statement.invoke({"param1": 13})

    def test_simple_flow(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = [
                ResultDefinition(result = 13),
                ResultDefinition(result = 29),
                CallDefinition(call = "some_function", params = {"param1": "{output0}", "param2": "{output1}"}),
                ResultDefinition(result = "{param1} is {output2}")
            ],
            tool_factory= lambda tool_ref, config: test_tool_logic
        )
        assert "Meaning of live is 42" == statement.invoke({"param1": "Meaning of live"})

    def test_match(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = MatchDefinition(
                match = "{param1}",
                trim = True,
                matchers = [
                    MatchClauseDefinition(
                        case = "Meaning of life",
                        then = ResultDefinition(result = 42)
                    ),
                    MatchClauseDefinition(
                        pattern = "[0-9]+",
                        then = ResultDefinition(result = "number")
                    ),
                    MatchClauseDefinition(
                        pattern = "https?://([^/]+)/?.*",
                        then = ResultDefinition(result = "an URL pointing to {match0}")
                    )
                ],
                default = ResultDefinition(result = -1)
            ),
            tool_factory= no_tool_lookup
        )
        self.assertEqual(42, statement.invoke({"param1": "Meaning of life"}))
        self.assertEqual("number", statement.invoke({"param1": "100"}))
        self.assertEqual("an URL pointing to www.google.com", statement.invoke({"param1": "https://www.google.com/"}))
        self.assertEqual(-1, statement.invoke({"param1": "Meaning of live is 42"}))
        self.assertEqual(-1, statement.invoke({"param1": ""}))
        self.assertEqual(-1, statement.invoke({"param1": {}}))
