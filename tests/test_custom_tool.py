import unittest

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from llm_workers.api import WorkerNotification
from llm_workers.config import CustomToolParamsDefinition, CallDefinition, ResultDefinition, \
    MatchDefinition, MatchClauseDefinition, WorkersConfig, CustomToolDefinition
from llm_workers.expressions import EvaluationContext, JsonExpression, StringExpression
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.tools.custom_tool import create_statement_from_model, build_custom_tool
from llm_workers.worker_utils import call_tool
from tests.mocks import StubWorkersContext


def get_stream_result(statement, context):
    """Helper to extract non-notification result from statement._stream()"""
    return next(chunk for chunk in statement._stream(EvaluationContext(context), None, {})
                if not isinstance(chunk, WorkerNotification))


@tool
def test_tool_logic(param1: int, param2: int) -> int:
    """Sum two parameters"""
    return param1 + param2


class TestStatements(unittest.TestCase):

    def test_return_string(self):
        statement = create_statement_from_model(
            model = ResultDefinition(result=JsonExpression("${param1} is 42")),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "Meaning of life"})
        self.assertEqual("Meaning of life is 42", result)

    def test_return_json(self):
        statement = create_statement_from_model(
            model = ResultDefinition(result=JsonExpression({"inner": "${param1} is 42"})),
            context=StubWorkersContext()
        )
        self.assertEqual({"inner": "Meaning of life is 42"}, get_stream_result(statement, {"param1": "Meaning of life"}))

    def test_simple_call(self):
        statement = create_statement_from_model(
            model = CallDefinition(call = "some_function", params = JsonExpression({"param1": "${param1}", "param2": 29})),
            context=StubWorkersContext(tools={"some_function": test_tool_logic})
        )
        assert 42 == get_stream_result(statement, {"param1": 13})

    def test_simple_flow(self):
        statement = create_statement_from_model(
            model = [
                ResultDefinition(result = JsonExpression(13)),
                ResultDefinition(result = JsonExpression(29)),
                CallDefinition(call = "some_function", params = JsonExpression({"param1": "${output0}", "param2": "${output1}"})),
                ResultDefinition(result = JsonExpression("${param1} is ${output2}"))
            ],
            context=StubWorkersContext(tools={"some_function": test_tool_logic})
        )
        assert "Meaning of live is 42" == get_stream_result(statement, {"param1": "Meaning of live"})

    def test_match(self):
        statement = create_statement_from_model(
            model = MatchDefinition(
                match = StringExpression("${param1}"),
                trim = True,
                matchers = [
                    MatchClauseDefinition(
                        case = "Meaning of life",
                        then = ResultDefinition(result = JsonExpression(42))
                    ),
                    MatchClauseDefinition(
                        pattern = "[0-9]+",
                        then = ResultDefinition(result = JsonExpression("number"))
                    ),
                    MatchClauseDefinition(
                        pattern = "https?://([^/]+)/?.*",
                        then = ResultDefinition(result = JsonExpression("an URL pointing to ${match[0]}"))
                    )
                ],
                default = ResultDefinition(result = JsonExpression(-1))
            ),
            context=StubWorkersContext()
        )
        self.assertEqual(42, get_stream_result(statement, {"param1": "Meaning of life"}))
        self.assertEqual("number", get_stream_result(statement, {"param1": "100"}))
        self.assertEqual("an URL pointing to www.google.com", get_stream_result(statement, {"param1": "https://www.google.com/"}))
        self.assertEqual(-1, get_stream_result(statement, {"param1": "Meaning of live is 42"}))
        self.assertEqual(-1, get_stream_result(statement, {"param1": ""}))
        self.assertEqual(-1, get_stream_result(statement, {"param1": {}}))


class TestSharedContentIntegration(unittest.TestCase):
    def test_custom_tool_with_shared_content(self):
        # Create a config with shared data
        config = WorkersConfig(
            shared=JsonExpression({
                "prompts": {
                    "test": "Yada-yada-yada"
                }
            }
        ))

        # Create mock context
        context = StubWorkersContext(config=config)
        
        # Define a custom tool that uses shared content
        tool_definition = CustomToolDefinition(
            name="demo_shared_access",
            description="Demo tool using shared content",
            input=[
                CustomToolParamsDefinition(name="query", description="Search query", type="str")
            ],
            body=ResultDefinition(result=JsonExpression("Query ${query} returned ${shared.prompts.test}"))
        )

        # Build the custom tool
        tool = build_custom_tool(tool_definition, context)
        
        # Test the tool
        result = tool.invoke({"query": "test_search"})
        self.assertEqual("Query test_search returned Yada-yada-yada", result)


class TestDynamicKeyResolution(unittest.TestCase):
    def test_dict_key_resolution(self):
        """Test resolving dictionary keys with result statement."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression({"json_schema": "schema_value", "other": "other_value"}),
                key=StringExpression("json_schema")
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "test"})
        self.assertEqual(result, "schema_value")

    def test_dict_key_resolution_with_default(self):
        """Test resolving dictionary keys with default value."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression({"json_schema": "schema_value", "other": "other_value"}),
                key=StringExpression("missing_key"),
                default=JsonExpression("default_value")
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "test"})
        self.assertEqual(result, "default_value")

    def test_list_key_resolution(self):
        """Test resolving list indices with result statement."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression(["first", "second", "third"]),
                key=StringExpression("1")
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "test"})
        self.assertEqual(result, "second")

    def test_list_key_resolution_with_default(self):
        """Test resolving list indices with default value."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression(["first", "second"]),
                key=StringExpression("5"),
                default=JsonExpression("default_value")
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "test"})
        self.assertEqual(result, "default_value")

    def test_dynamic_key_resolution(self):
        """Test dynamic key resolution using template variables."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression("${data}"),
                key=StringExpression("${key_name}"),
                default=JsonExpression("not_found")
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {
            "key_name": "target_key",
            "data": {"target_key": "found_value", "other": "other_value"}
        })
        self.assertEqual(result, "found_value")

    def test_no_key_resolution(self):
        """Test that result statement works normally without key parameter."""
        statement = create_statement_from_model(
            model=ResultDefinition(
                result=JsonExpression({"json_schema": "schema_value", "other": "other_value"})
            ),
            context=StubWorkersContext()
        )
        result = get_stream_result(statement, {"param1": "test"})
        self.assertEqual(result, {"json_schema": "schema_value", "other": "other_value"})


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
            body=CallDefinition(
                call="inner_sum_tool",
                params=JsonExpression({"a": "${value1}", "b": "${value2}"})
            )
        )

        custom_tool = build_custom_tool(tool_definition, context)

        # Call the custom tool using call_tool
        token_tracker = CompositeTokenUsageTracker()
        config = RunnableConfig()

        chunks = list(call_tool(
            tool=custom_tool,
            input={"value1": 13, "value2": 29},
            evaluation_context=EvaluationContext(),
            token_tracker=token_tracker,
            config=config,
            kwargs={}
        ))

        # Separate notifications from results
        notifications = [c for c in chunks if isinstance(c, WorkerNotification)]
        results = [c for c in chunks if not isinstance(c, WorkerNotification)]

        # Verify result
        self.assertEqual(len(results), 1, f"Expected 1 result but got {len(results)}")
        self.assertEqual(results[0], 42, f"Expected result 42 but got {results[0]}")

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
