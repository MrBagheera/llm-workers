import unittest

import yaml
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from llm_workers.config import CustomToolParamsDefinition, CallDefinition, EvalDefinition, \
    MatchDefinition, WorkersConfig, CustomToolDefinition
from llm_workers.expressions import EvaluationContext, JsonExpression
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.tools.custom_tool import create_statement_from_model, build_custom_tool
from llm_workers.worker_utils import call_tool, split_result_and_notifications
from tests.mocks import StubWorkersContext


@tool
def test_tool_logic(param1: int, param2: int) -> int:
    """Sum two parameters"""
    return param1 + param2


class TestStatements(unittest.TestCase):

    def test_return_string(self):
        statement = create_statement_from_model(
            model = EvalDefinition(eval=JsonExpression("${param1} is 42")),
            context=StubWorkersContext()
        )
        context = {"param1": "Meaning of life"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual("Meaning of life is 42", result)

    def test_return_json(self):
        statement = create_statement_from_model(
            model = EvalDefinition(eval=JsonExpression({"inner": "${param1} is 42"})),
            context=StubWorkersContext()
        )
        context = {"param1": "Meaning of life"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        self.assertEqual({"inner": "Meaning of life is 42"}, split_result_and_notifications(generator)[0])

    def test_simple_call(self):
        statement = create_statement_from_model(
            model = CallDefinition(call = "some_function", params = JsonExpression({"param1": "${param1}", "param2": 29})),
            context=StubWorkersContext(tools={"some_function": test_tool_logic})
        )
        context = {"param1": 13}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        assert 42 == split_result_and_notifications(generator)[0]

    def test_simple_flow(self):
        statement = create_statement_from_model(
            model = [
                EvalDefinition(eval = JsonExpression(13), store_as="output0"),
                EvalDefinition(eval = JsonExpression(29), store_as="output1"),
                CallDefinition(call = "some_function", params = JsonExpression({"param1": "${output0}", "param2": "${output1}"})),
                EvalDefinition(eval = JsonExpression("${param1} is ${_}"))
            ],
            context=StubWorkersContext(tools={"some_function": test_tool_logic})
        )
        context = {"param1": "Meaning of live"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        assert "Meaning of live is 42" == split_result_and_notifications(generator)[0]

    def test_match(self):
        statement = create_statement_from_model(
            model = MatchDefinition.model_validate(yaml.safe_load("""
            match: ${param1}
            trim: true
            matchers:
              - case: Meaning of life
                then:
                  eval: 42
              - pattern: '[0-9]+'
                then:
                  eval: 'number'
              - pattern: 'https?://([^/]+)/?.*'
                then:
                  eval: 'an URL pointing to ${_match_groups[0]}'
            default:
              eval: -1
            """)),
            context=StubWorkersContext()
        )
        context = {"param1": "Meaning of life"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        self.assertEqual(42, split_result_and_notifications(generator)[0])
        context1 = {"param1": "100"}
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), token_tracker=None, config=None)
        self.assertEqual("number", split_result_and_notifications(generator1)[0])
        context2 = {"param1": "https://www.google.com/"}
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), token_tracker=None, config=None)
        self.assertEqual("an URL pointing to www.google.com", split_result_and_notifications(generator2)[0])
        context3 = {"param1": "Meaning of live is 42"}
        generator3 = statement.yield_notifications_and_result(EvaluationContext(context3), token_tracker=None, config=None)
        self.assertEqual(-1, split_result_and_notifications(generator3)[0])
        context4 = {"param1": ""}
        generator4 = statement.yield_notifications_and_result(EvaluationContext(context4), token_tracker=None, config=None)
        self.assertEqual(-1, split_result_and_notifications(generator4)[0])
        context5 = {"param1": {}}
        generator5 = statement.yield_notifications_and_result(EvaluationContext(context5), token_tracker=None, config=None)
        self.assertEqual(-1, split_result_and_notifications(generator5)[0])


class TestSharedContentIntegration(unittest.TestCase):
    def test_custom_tool_with_shared_content(self):
        # Create a config with shared data
        from llm_workers.config import SharedConfig
        config = WorkersConfig(
            shared=SharedConfig(data=JsonExpression({
                "prompts": {
                    "test": "Yada-yada-yada"
                }
            }))
        )

        # Create mock context
        context = StubWorkersContext(config=config)
        
        # Define a custom tool that uses shared content
        tool_definition = CustomToolDefinition(
            name="demo_shared_access",
            description="Demo tool using shared content",
            input=[
                CustomToolParamsDefinition(name="query", description="Search query", type="str")
            ],
            do=EvalDefinition(eval=JsonExpression("Query ${query} returned ${shared.prompts.test}"))
        )

        # Build the custom tool
        tool = build_custom_tool(tool_definition, context)
        
        # Test the tool
        result = tool.invoke({"query": "test_search"})
        self.assertEqual("Query test_search returned Yada-yada-yada", result)


class TestEvalStatementMigrationPatterns(unittest.TestCase):
    """Test examples showing how to migrate from result+key+default to eval with expressions."""

    def test_dict_get_with_default(self):
        """Migration: Use dict.get() for dictionary access with default."""
        # Old: result: "${data}", key: "json_schema", default: "default_value"
        # New: eval: "${data.get('json_schema', 'default_value')}"
        statement = create_statement_from_model(
            model=EvalDefinition(
                eval=JsonExpression("${data.get('json_schema', 'default_value')}")
            ),
            context=StubWorkersContext()
        )

        # Test with existing key
        context = {
            "data": {"json_schema": "schema_value", "other": "other_value"}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "schema_value")

        # Test with missing key
        context1 = {
            "data": {"other": "other_value"}
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), token_tracker=None, config=None)
        result = split_result_and_notifications(generator1)[0]
        self.assertEqual(result, "default_value")

    def test_dynamic_key_from_variable(self):
        """Migration: Use dict.get() with dynamic key variable."""
        # Old: result: "${data}", key: "${key_name}", default: "not_found"
        # New: eval: "${data.get(key_name, 'not_found')}"
        statement = create_statement_from_model(
            model=EvalDefinition(
                eval=JsonExpression("${data.get(key_name, 'not_found')}")
            ),
            context=StubWorkersContext()
        )
        context = {
            "key_name": "target_key",
            "data": {"target_key": "found_value", "other": "other_value"}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "found_value")

    def test_list_index_with_bounds_check(self):
        """Migration: Use conditional expression for list access with default."""
        # Old: result: "${items}", key: "${index}", default: "out_of_bounds"
        # New: eval: "${items[index] if 0 <= index < len(items) else 'out_of_bounds'}"
        statement = create_statement_from_model(
            model=EvalDefinition(
                eval=JsonExpression("${items[index] if 0 <= index < len(items) else 'out_of_bounds'}")
            ),
            context=StubWorkersContext()
        )

        # Test valid index
        context = {
            "items": ["first", "second", "third"],
            "index": 1
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "second")

        # Test out of bounds
        context1 = {
            "items": ["first", "second"],
            "index": 5
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), token_tracker=None, config=None)
        result = split_result_and_notifications(generator1)[0]
        self.assertEqual(result, "out_of_bounds")

    def test_simple_list_index(self):
        """Migration: Direct list indexing when bounds are guaranteed."""
        # Old: result: ["a", "b", "c"], key: "1"
        # New: eval: "${items[1]}" where items is provided
        statement = create_statement_from_model(
            model=EvalDefinition(
                eval=JsonExpression("${items[1]}")
            ),
            context=StubWorkersContext()
        )
        context = {"items": ["a", "b", "c"]}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "b")

    def test_bracket_notation_for_nested_access(self):
        """Migration: Use bracket notation for nested dictionary access."""
        # Old: result: "${data}", key: "${field}", default: "N/A"
        # New: eval: "${data['level1']['level2'] if 'level1' in data and 'level2' in data['level1'] else 'N/A'}"
        statement = create_statement_from_model(
            model=EvalDefinition(
                eval=JsonExpression("${data['level1']['level2'] if 'level1' in data and 'level2' in data['level1'] else 'N/A'}")
            ),
            context=StubWorkersContext()
        )

        # Test existing nested value
        context = {
            "data": {"level1": {"level2": "found"}}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), token_tracker=None, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "found")

        # Test missing nested value
        context1 = {
            "data": {"level1": {}}
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), token_tracker=None, config=None)
        result = split_result_and_notifications(generator1)[0]
        self.assertEqual(result, "N/A")


class TestHierarchicalToolCalling(unittest.TestCase):
    """Test hierarchical tool calling where CustomTool calls another tool."""

    def test_hierarchical_tool_calling_notifications(self):
        """Test that CustomTool calling another tool generates proper notifications with run_id hierarchy."""

        # Create a simple tool that will be called by the CustomTool
        @tool
        def inner_sum_tool(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        # Create a CustomTool that calls inner_sum_tool
        context = StubWorkersContext(tools={"inner_sum_tool": inner_sum_tool})

        tool_definition = CustomToolDefinition(
            name="outer_calculator",
            description="Tool that calls another tool to calculate",
            input=[
                CustomToolParamsDefinition(name="value1", description="First value", type="int"),
                CustomToolParamsDefinition(name="value2", description="Second value", type="int")
            ],
            do=CallDefinition(
                call="inner_sum_tool",
                params=JsonExpression({"a": "${value1}", "b": "${value2}"})
            )
        )

        custom_tool = build_custom_tool(tool_definition, context)

        # Call the custom tool using call_tool
        token_tracker = CompositeTokenUsageTracker()
        config = RunnableConfig()

        generator = call_tool(
            tool=custom_tool,
            input={"value1": 13, "value2": 29},
            evaluation_context=EvaluationContext(),
            token_tracker=token_tracker,
            config=config,
            kwargs={}
        )

        # Separate notifications from results
        result, notifications = split_result_and_notifications(generator)

        # Verify result
        self.assertEqual(result, 42)

        # Verify notifications structure
        # Should have: outer_tool_start, inner_tool_start, inner_tool_end, outer_tool_end
        self.assertEqual(len(notifications), 4, f"Expected 4 notifications but got {len(notifications)}")

        outer_start = notifications[0]
        inner_start = notifications[1]
        inner_end = notifications[2]
        outer_end = notifications[3]

        # Verify types
        self.assertEqual(outer_start.type, 'tool_start', "First notification should be outer tool_start")
        self.assertEqual(inner_start.type, 'tool_start', "Second notification should be inner tool_start")
        self.assertEqual(inner_end.type, 'tool_end', "Third notification should be inner tool_end")
        self.assertEqual(outer_end.type, 'tool_end', "Fourth notification should be outer tool_end")

        # Verify run_id hierarchy
        self.assertIsNotNone(outer_start.run_id, "Outer tool should have a run_id")
        self.assertIsNone(outer_start.parent_run_id, "Outer tool should have no parent (top-level call)")

        self.assertIsNotNone(inner_start.run_id, "Inner tool should have a run_id")
        self.assertEqual(inner_start.parent_run_id, outer_start.run_id,
                        "Inner tool's parent_run_id should match outer tool's run_id")

        # Verify run_ids match for tool_start and tool_end
        self.assertEqual(inner_end.run_id, inner_start.run_id,
                        "Inner tool_end run_id should match tool_start run_id")
        self.assertEqual(outer_end.run_id, outer_start.run_id,
                        "Outer tool_end run_id should match tool_start run_id")

        # Verify tool names in notification text
        self.assertIsNotNone(outer_start.text, "Outer tool_start should have text")
        self.assertIsNotNone(inner_start.text, "Inner tool_start should have text")
        self.assertIn("outer_calculator", outer_start.text,
                     "Outer tool name should appear in notification text")
        self.assertIn("inner_sum_tool", inner_start.text,
                     "Inner tool name should appear in notification text")
