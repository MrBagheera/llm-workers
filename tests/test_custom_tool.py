import unittest

from langchain_core.tools import tool

from llm_workers.tools.custom_tool import TemplateHelper, create_statement_from_model, build_custom_tool
from llm_workers.config import CustomToolParamsDefinition, CallDefinition, ResultDefinition, \
    MatchDefinition, MatchClauseDefinition, ToolDefinition, WorkersConfig


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

    def test_nested_element_references(self):
        helper = TemplateHelper.from_param_definitions(
            params = [
                CustomToolParamsDefinition(name = "param_dict", description = "A dictionary parameter", type = "object"),
                CustomToolParamsDefinition(name = "param_list", description = "A list parameter", type = "array"),
            ],
            target_params = {
                "dict_access": "{param_dict[key1]}",
                "list_access": "{param_list[0]}",
                "nested_access": "Combined: {param_dict[nested][value]} and {param_list[1]}"
            },
        )
        result = helper.render({
            "param_dict": {
                "key1": "hello",
                "nested": {"value": "world"}
            },
            "param_list": ["first", "second", "third"]
        })
        self.assertDictEqual(result, {
            "dict_access": "hello",
            "list_access": "first",
            "nested_access": "Combined: world and second"
        })

    def test_shared_content_references(self):
        helper = TemplateHelper.from_valid_template_vars(
            valid_template_vars = ["param1", "shared"],
            target_params = {
                "shared_access": "{shared[prompts][test]}",
                "mixed_access": "Query {param1} returned {shared[prompts][test]}"
            }
        )
        result = helper.render({
            "param1": "search_term",
            "shared": {
                "prompts": {
                    "test": "Yada-yada-yada"
                }
            }
        })
        self.assertDictEqual(result, {
            "shared_access": "Yada-yada-yada",
            "mixed_access": "Query search_term returned Yada-yada-yada"
        })


@tool
def test_tool_logic(param1: int, param2: int) -> int:
    """Sum two parameters"""
    return param1 + param2

def no_tool_lookup(tool_ref, config):
    raise ValueError(f"Unexpected tool lookup: {tool_ref}")

class MockContextNoTools:
    def __init__(self):
        pass
    
    def get_tool(self, tool_name: str):
        raise ValueError(f"Unexpected tool lookup: {tool_name}")

class MockContextWithTestTool:
    def __init__(self):
        pass
    
    def get_tool(self, tool_name: str):
        if tool_name == "some_function":
            return test_tool_logic
        raise ValueError(f"Unexpected tool lookup: {tool_name}")

class TestStatements(unittest.TestCase):

    def test_return_string(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = ResultDefinition(result="{param1} is 42"),
            context=MockContextNoTools()
        )
        self.assertEqual("Meaning of life is 42", statement.invoke({"param1": "Meaning of life"}))

    def test_return_json(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = ResultDefinition(result={"inner": "{param1} is 42"}),
            context=MockContextNoTools()
        )
        self.assertEqual({"inner": "Meaning of life is 42"}, statement.invoke({"param1": "Meaning of life"}))

    def test_simple_call(self):
        statement = create_statement_from_model(
            valid_template_vars = ["param1"],
            model = CallDefinition(call = "some_function", params = {"param1": "{param1}", "param2": 29}),
            context=MockContextWithTestTool()
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
            context=MockContextWithTestTool()
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
            context=MockContextNoTools()
        )
        self.assertEqual(42, statement.invoke({"param1": "Meaning of life"}))
        self.assertEqual("number", statement.invoke({"param1": "100"}))
        self.assertEqual("an URL pointing to www.google.com", statement.invoke({"param1": "https://www.google.com/"}))
        self.assertEqual(-1, statement.invoke({"param1": "Meaning of live is 42"}))
        self.assertEqual(-1, statement.invoke({"param1": ""}))
        self.assertEqual(-1, statement.invoke({"param1": {}}))


# Mock context for testing
class MockContext:
    def __init__(self, config):
        self.config = config

    def get_tool(self, tool_name: str):
        raise ValueError(f"Tool {tool_name} not found")


class TestSharedContentIntegration(unittest.TestCase):
    def test_custom_tool_with_shared_content(self):
        # Create a config with shared data
        config = WorkersConfig(
            shared={
                "prompts": {
                    "test": "Yada-yada-yada"
                }
            }
        )
        
        # Create mock context
        context = MockContext(config)
        
        # Define a custom tool that uses shared content
        tool_definition = ToolDefinition(
            name="demo_shared_access",
            description="Demo tool using shared content",
            input=[
                CustomToolParamsDefinition(name="query", description="Search query", type="str")
            ],
            body=ResultDefinition(result="Query {query} returned {shared[prompts][test]}")
        )
        
        # Build the custom tool
        tool = build_custom_tool(tool_definition, context)
        
        # Test the tool
        result = tool.invoke({"query": "test_search"})
        self.assertEqual(result, "Query test_search returned Yada-yada-yada")
