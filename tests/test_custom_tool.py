import unittest

import yaml
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from llm_workers.config import CustomToolParamsDefinition, CallDefinition, EvalDefinition, \
    IfDefinition, StarlarkDefinition, WorkersConfig, CustomToolDefinition
from llm_workers.expressions import EvaluationContext, JsonExpression
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.tools.custom_tool import create_statement_from_model, build_custom_tool
from llm_workers.worker_utils import call_tool, split_result_and_notifications
from tests.mocks import StubWorkersContext


@tool
def test_tool_logic(param1: int, param2: int) -> int:
    """Sum two parameters"""
    return param1 + param2

_token_tracker = CompositeTokenUsageTracker()

class TestStatements(unittest.TestCase):

    def test_return_string(self):
        statement = create_statement_from_model(
            model = EvalDefinition(eval=JsonExpression("${param1} is 42")),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"param1": "Meaning of life"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual("Meaning of life is 42", result)

    def test_return_json(self):
        statement = create_statement_from_model(
            model = EvalDefinition(eval=JsonExpression({"inner": "${param1} is 42"})),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"param1": "Meaning of life"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual({"inner": "Meaning of life is 42"}, split_result_and_notifications(generator)[0])

    def test_simple_call(self):
        statement = create_statement_from_model(
            model = CallDefinition(call = "some_function", params = JsonExpression({"param1": "${param1}", "param2": 29})),
            context=StubWorkersContext(tools={"some_function": test_tool_logic}),
            local_tools={}
        )
        context = {"param1": 13}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        assert 42 == split_result_and_notifications(generator)[0]

    def test_simple_flow(self):
        statement = create_statement_from_model(
            model = [
                EvalDefinition(eval = JsonExpression(13), store_as="output0"),
                EvalDefinition(eval = JsonExpression(29), store_as="output1"),
                CallDefinition(call = "some_function", params = JsonExpression({"param1": "${output0}", "param2": "${output1}"})),
                EvalDefinition(eval = JsonExpression("${param1} is ${_}"))
            ],
            context=StubWorkersContext(tools={"some_function": test_tool_logic}),
            local_tools={}
        )
        context = {"param1": "Meaning of live"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        assert "Meaning of live is 42" == split_result_and_notifications(generator)[0]

class TestIfStatement(unittest.TestCase):
    """Test if-then-else statement functionality."""

    def test_if_true_with_then_only(self):
        """Test if condition true, no else clause."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${flag}"
            then:
              eval: "executed"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"flag": True}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("executed", split_result_and_notifications(generator)[0])

    def test_if_false_with_then_only_returns_none(self):
        """Test if condition false, no else clause (returns None)."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${flag}"
            then:
              eval: "executed"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"flag": False}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertIsNone(split_result_and_notifications(generator)[0])

    def test_if_true_with_else(self):
        """Test if condition true with else clause (else not executed)."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${flag}"
            then:
              eval: "then branch"
            else:
              eval: "else branch"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"flag": True}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("then branch", split_result_and_notifications(generator)[0])

    def test_if_false_with_else(self):
        """Test if condition false, else clause executed."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${flag}"
            then:
              eval: "then branch"
            else:
              eval: "else branch"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"flag": False}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("else branch", split_result_and_notifications(generator)[0])

    def test_if_with_store_as(self):
        """Test storing result in variable."""
        statement = create_statement_from_model(
            model=[
                IfDefinition.model_validate(yaml.safe_load("""
                if: "${flag}"
                then:
                  eval: "result_value"
                store_as: result
                """)),
                EvalDefinition(eval=JsonExpression("Got: ${result}"))
            ],
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"flag": True}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("Got: result_value", split_result_and_notifications(generator)[0])

    def test_if_truthiness_empty_string(self):
        """Test empty string evaluates to False."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": ""}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("falsy", split_result_and_notifications(generator)[0])

    def test_if_truthiness_non_empty_string(self):
        """Test non-empty string evaluates to True."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": "hello"}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("truthy", split_result_and_notifications(generator)[0])

    def test_if_truthiness_zero(self):
        """Test zero evaluates to False."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": 0}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("falsy", split_result_and_notifications(generator)[0])

    def test_if_truthiness_non_zero_number(self):
        """Test non-zero number evaluates to True."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": 42}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("truthy", split_result_and_notifications(generator)[0])

    def test_if_truthiness_empty_list(self):
        """Test empty list evaluates to False."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": []}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("falsy", split_result_and_notifications(generator)[0])

    def test_if_truthiness_non_empty_list(self):
        """Test non-empty list evaluates to True."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${value}"
            then:
              eval: "truthy"
            else:
              eval: "falsy"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"value": [1, 2, 3]}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("truthy", split_result_and_notifications(generator)[0])

    def test_if_with_boolean_expression(self):
        """Test complex boolean expression: ${x > 5 and y < 10}"""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${x > 5 and y < 10}"
            then:
              eval: "condition met"
            else:
              eval: "condition not met"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"x": 7, "y": 8}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("condition met", split_result_and_notifications(generator)[0])

        context2 = {"x": 3, "y": 8}
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), _token_tracker, config=None)
        self.assertEqual("condition not met", split_result_and_notifications(generator2)[0])

    def test_if_with_membership_test(self):
        """Test membership: ${key in dict}"""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${movie_title in metacritic_stub_data}"
            then:
              eval: "${metacritic_stub_data[movie_title]}"
            else:
              eval: "not found"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {
            "movie_title": "Soul",
            "metacritic_stub_data": {"Soul": "83", "The Incredibles": "90"}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual("83", split_result_and_notifications(generator)[0])

        context2 = {
            "movie_title": "Unknown Movie",
            "metacritic_stub_data": {"Soul": "83", "The Incredibles": "90"}
        }
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), _token_tracker, config=None)
        self.assertEqual("not found", split_result_and_notifications(generator2)[0])

    def test_if_nested_statements(self):
        """Test if with nested if-then-else in branches."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${level1}"
            then:
              if: "${level2}"
              then:
                eval: "both true"
              else:
                eval: "only level1 true"
            else:
              eval: "level1 false"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context1 = {"level1": True, "level2": True}
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), _token_tracker, config=None)
        self.assertEqual("both true", split_result_and_notifications(generator1)[0])

        context2 = {"level1": True, "level2": False}
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), _token_tracker, config=None)
        self.assertEqual("only level1 true", split_result_and_notifications(generator2)[0])

        context3 = {"level1": False, "level2": True}
        generator3 = statement.yield_notifications_and_result(EvaluationContext(context3), _token_tracker, config=None)
        self.assertEqual("level1 false", split_result_and_notifications(generator3)[0])

    def test_if_with_call_statement_in_branch(self):
        """Test calling tools in then/else branches."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${should_call}"
            then:
              call: test_tool_logic
              params:
                param1: "${value1}"
                param2: "${value2}"
            else:
              eval: "skipped"
            """)),
            context=StubWorkersContext(tools={"test_tool_logic": test_tool_logic}),
            local_tools={}
        )
        context = {"should_call": True, "value1": 13, "value2": 29}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual(42, split_result_and_notifications(generator)[0])

        context2 = {"should_call": False, "value1": 13, "value2": 29}
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), _token_tracker, config=None)
        self.assertEqual("skipped", split_result_and_notifications(generator2)[0])

    def test_if_with_flow_in_branch(self):
        """Test multiple statements in then/else branches."""
        statement = create_statement_from_model(
            model=IfDefinition.model_validate(yaml.safe_load("""
            if: "${execute_flow}"
            then:
              - eval: "${base}"
                store_as: step1
              - eval: "${step1 + 10}"
                store_as: step2
              - eval: "${step2}"
            else:
              eval: "flow skipped"
            """)),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"execute_flow": True, "base": 5}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        self.assertEqual(15, split_result_and_notifications(generator)[0])

        context2 = {"execute_flow": False, "base": 5}
        generator2 = statement.yield_notifications_and_result(EvaluationContext(context2), _token_tracker, config=None)
        self.assertEqual("flow skipped", split_result_and_notifications(generator2)[0])


class TestSharedContentIntegration(unittest.TestCase):
    def test_custom_tool_with_shared_content(self):
        # Create a config with shared data
        from llm_workers.config import SharedSectionConfig
        config = WorkersConfig(
            shared=SharedSectionConfig(data={
                "prompts": JsonExpression({
                    "test": "Yada-yada-yada"
                })
            })
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
            do=EvalDefinition(eval=JsonExpression("Query ${query} returned ${prompts.test}"))
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
            context=StubWorkersContext(),
            local_tools={}
        )

        # Test with existing key
        context = {
            "data": {"json_schema": "schema_value", "other": "other_value"}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "schema_value")

        # Test with missing key
        context1 = {
            "data": {"other": "other_value"}
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), _token_tracker, config=None)
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
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {
            "key_name": "target_key",
            "data": {"target_key": "found_value", "other": "other_value"}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
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
            context=StubWorkersContext(),
            local_tools={}
        )

        # Test valid index
        context = {
            "items": ["first", "second", "third"],
            "index": 1
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "second")

        # Test out of bounds
        context1 = {
            "items": ["first", "second"],
            "index": 5
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), _token_tracker, config=None)
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
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"items": ["a", "b", "c"]}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
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
            context=StubWorkersContext(),
            local_tools={}
        )

        # Test existing nested value
        context = {
            "data": {"level1": {"level2": "found"}}
        }
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(result, "found")

        # Test missing nested value
        context1 = {
            "data": {"level1": {}}
        }
        generator1 = statement.yield_notifications_and_result(EvaluationContext(context1), _token_tracker, config=None)
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


class TestStarlarkStatement(unittest.TestCase):
    """Test Starlark statement functionality."""

    def test_simple_starlark_with_result_variable(self):
        """Test Starlark script returning via result variable."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="result = 1 + 2"),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(3, result)

    def test_starlark_with_run_function(self):
        """Test Starlark script returning via run() function."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="""
def run():
    return 1 + 2
"""),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(3, result)

    def test_starlark_with_variable_access(self):
        """Test accessing input variables from Starlark."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="result = x + y"),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"x": 10, "y": 20}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(30, result)

    def test_starlark_calling_tool(self):
        """Test calling a tool from Starlark script."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="result = test_tool_logic(param1=5, param2=10)"),
            context=StubWorkersContext(tools={"test_tool_logic": test_tool_logic}),
            local_tools={"test_tool_logic": test_tool_logic}
        )
        context = {}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(15, result)

    def test_starlark_calling_multiple_tools(self):
        """Test calling multiple tools in sequence."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="""
a = test_tool_logic(param1=5, param2=10)
result = test_tool_logic(param1=a, param2=20)
"""),
            context=StubWorkersContext(tools={"test_tool_logic": test_tool_logic}),
            local_tools={"test_tool_logic": test_tool_logic}
        )
        context = {}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(35, result)  # 5 + 10 = 15, 15 + 20 = 35

    def test_starlark_with_conditionals(self):
        """Test Starlark if/else logic."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="""
if x > 10:
    result = "large"
else:
    result = "small"
"""),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {"x": 15}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual("large", result)

    def test_starlark_with_loops(self):
        """Test Starlark for loops."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="""
total = 0
for i in [1, 2, 3, 4, 5]:
    total = total + i
result = total
"""),
            context=StubWorkersContext(),
            local_tools={}
        )
        context = {}
        generator = statement.yield_notifications_and_result(EvaluationContext(context), _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(15, result)

    def test_starlark_with_store_as(self):
        """Test storing Starlark result in variable."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="result = 42", store_as="answer"),
            context=StubWorkersContext(),
            local_tools={}
        )
        context_obj = EvaluationContext({})
        generator = statement.yield_notifications_and_result(context_obj, _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(42, result)
        self.assertEqual(42, context_obj.get("answer"))

    def test_starlark_error_handling(self):
        """Test error handling when Starlark script fails."""
        # Invalid syntax should raise error during initialization
        with self.assertRaises(SyntaxError):
            create_statement_from_model(
                model=StarlarkDefinition(starlark="while True: pass"),  # while loops not allowed
                context=StubWorkersContext(),
                local_tools={}
            )

    def test_starlark_with_parent_context(self):
        """Test accessing variables from parent evaluation context."""
        statement = create_statement_from_model(
            model=StarlarkDefinition(starlark="result = parent_var + child_var"),
            context=StubWorkersContext(),
            local_tools={}
        )
        parent_context = EvaluationContext({"parent_var": 100})
        child_context = EvaluationContext({"child_var": 23}, parent=parent_context)
        generator = statement.yield_notifications_and_result(child_context, _token_tracker, config=None)
        result = split_result_and_notifications(generator)[0]
        self.assertEqual(123, result)
