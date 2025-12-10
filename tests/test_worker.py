import unittest
from unittest.mock import Mock

from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool

from llm_workers.core.config import BaseLLMConfig
from llm_workers.core.worker import Worker
from tests.mocks import MockInvokeLLM, StubWorkersContext, MockStreamLLM


class TestWorker(unittest.TestCase):

    def test_single_message_non_streaming(self):
        """Test basic non-streaming message processing."""
        # Setup: Create fake LLM with expectations
        input_message = HumanMessage(content="Hello, world!")

        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=AIMessage(content="Hi there!")
        )

        # Create mock context and worker
        context = StubWorkersContext(fake_llm)
        llm_config = BaseLLMConfig(model_ref="default")
        worker = Worker(llm_config, context)

        # Execute: Invoke worker with non-streaming
        result = worker.invoke([input_message], stream=False)

        # Verify: invoke() returns only BaseMessage instances
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], AIMessage)
        self.assertEqual(result[0].content, "Hi there!")

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_stream_non_streaming(self):
        """Test worker.stream() with stream=False returns both notifications and messages."""
        # Setup: Create fake LLM with expectations
        input_message = HumanMessage(content="Hello, world!")

        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=AIMessage(content="Hi there!")
        )

        # Create mock context and worker
        context = StubWorkersContext(fake_llm)
        llm_config = BaseLLMConfig(model_ref="default")
        worker = Worker(llm_config, context)

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify: stream() returns both notifications and messages
        from llm_workers.core.api import WorkerNotification

        # Expected order: thinking_start, thinking_end, AIMessage
        self.assertEqual(len(result), 3)

        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(result[2].content, "Hi there!")

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_stream_streaming(self):
        """Test worker.stream() with stream=True returns chunks and messages."""
        # Setup: Create fake LLM with expectations
        input_message = HumanMessage(content="Hello, world!")

        fake_llm = MockStreamLLM()
        fake_llm.expect_stream(
            input=[input_message],
            output=[
                AIMessageChunk(content="Hi"),
                AIMessageChunk(content=" there!")
            ]
        )

        # Create mock context and worker
        context = StubWorkersContext(fake_llm)
        llm_config = BaseLLMConfig(model_ref="default")
        worker = Worker(llm_config, context)

        # Execute: Stream worker with streaming (stream=True)
        result = []
        for chunk in worker.stream([input_message], stream=True):
            result.extend(chunk)

        # Verify: stream() returns notifications and message
        from llm_workers.core.api import WorkerNotification

        # Expected order: thinking_start, ai_output_chunk (Hi), ai_output_chunk ( there!), thinking_end, AIMessage
        self.assertEqual(len(result), 5)

        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'ai_output_chunk')
        self.assertEqual(result[1].text, "Hi")

        self.assertIsInstance(result[2], WorkerNotification)
        self.assertEqual(result[2].type, 'ai_output_chunk')
        self.assertEqual(result[2].text, " there!")

        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'thinking_end')

        self.assertIsInstance(result[4], AIMessage)
        self.assertEqual(result[4].content, "Hi there!")

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_single_standard_tool_call(self):
        """Test worker with a single standard (non-direct) tool call."""
        # Setup: Create a simple tool
        class SimpleTool(BaseTool):
            name: str = "simple_tool"
            description: str = "A simple test tool"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Processed: {input_str}"

        # Create tool with metadata
        tool = SimpleTool()
        tool.metadata = {'tool_definition': Mock(
            name='simple_tool',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Please use the tool")

        # First LLM response with tool call
        tool_call_response = AIMessage(
            content="I'll use the tool",
            tool_calls=[{
                'id': 'call_123',
                'name': 'simple_tool',
                'args': {'input_str': 'test input'}
            }]
        )

        # Second LLM response after tool execution
        final_response = AIMessage(content="Tool executed successfully")

        # Setup fake LLM with expectations
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        ).expect_invoke(
            input=[
                input_message,
                tool_call_response,
                ToolMessage(content='Processed: test input', tool_call_id='call_123', name='simple_tool')
            ],
            output=final_response
        )

        # Create mock context with tool
        context = StubWorkersContext(fake_llm)
        context._tools = {'simple_tool': tool}

        llm_config = BaseLLMConfig(model_ref="default", tools=['simple_tool'])
        worker = Worker(llm_config, context)
        worker._tools = {'simple_tool': tool}

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify the sequence of messages
        from llm_workers.core.api import WorkerNotification

        # Expected order:
        # 1. thinking_start
        # 2. thinking_end
        # 3. AIMessage with tool_call
        # 4. tool_start notification
        # 5. tool_end notification
        # 6. ToolMessage with result
        # 7. thinking_start (second iteration)
        # 8. thinking_end (second iteration)
        # 9. Final AIMessage

        self.assertEqual(len(result), 9)

        # First iteration - LLM returns tool call
        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 1)
        self.assertEqual(result[2].tool_calls[0]['name'], 'simple_tool')

        # Tool execution
        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'tool_start')
        self.assertEqual(result[3].text, 'Running tool simple_tool')

        self.assertIsInstance(result[4], WorkerNotification)
        self.assertEqual(result[4].type, 'tool_end')

        self.assertIsInstance(result[5], ToolMessage)
        self.assertEqual(result[5].content, 'Processed: test input')
        self.assertEqual(result[5].tool_call_id, 'call_123')
        self.assertEqual(result[5].name, 'simple_tool')

        # Second iteration - LLM returns final response
        self.assertIsInstance(result[6], WorkerNotification)
        self.assertEqual(result[6].type, 'thinking_start')

        self.assertIsInstance(result[7], WorkerNotification)
        self.assertEqual(result[7].type, 'thinking_end')

        self.assertIsInstance(result[8], AIMessage)
        self.assertEqual(result[8].content, 'Tool executed successfully')

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_single_direct_tool_call(self):
        """Test worker with a single direct-result tool call."""
        # Setup: Create a simple tool with return_direct=True
        class DirectTool(BaseTool):
            name: str = "direct_tool"
            description: str = "A tool that returns directly"
            return_direct: bool = True

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Direct result: {input_str}"

        # Create tool with metadata
        tool = DirectTool()
        tool.metadata = {'tool_definition': Mock(
            name='direct_tool',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Please use the direct tool")

        # LLM response with tool call (should only be called once)
        tool_call_response = AIMessage(
            content="I'll use the direct tool",
            tool_calls=[{
                'id': 'call_456',
                'name': 'direct_tool',
                'args': {'input_str': 'test input'}
            }]
        )

        # Setup fake LLM with expectations - only ONE call expected
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        )

        # Create mock context with tool
        context = StubWorkersContext(fake_llm)
        context._tools = {'direct_tool': tool}

        llm_config = BaseLLMConfig(model_ref="default", tools=['direct_tool'])
        worker = Worker(llm_config, context)
        worker._tools = {'direct_tool': tool}

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify the sequence of messages
        from llm_workers.core.api import WorkerNotification

        # Expected order:
        # 1. thinking_start
        # 2. thinking_end
        # 3. AIMessage with tool_call
        # 4. tool_start notification
        # 5. tool_end notification
        # 6. ToolMessage (stub message about direct result)
        # 7. AIMessage with direct result (from tool execution)

        self.assertEqual(len(result), 7)

        # First iteration - LLM returns tool call
        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 1)
        self.assertEqual(result[2].tool_calls[0]['name'], 'direct_tool')

        # Tool execution
        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'tool_start')

        self.assertIsInstance(result[4], WorkerNotification)
        self.assertEqual(result[4].type, 'tool_end')

        # Stub ToolMessage (tells LLM result was shown directly)
        self.assertIsInstance(result[5], ToolMessage)
        self.assertEqual(result[5].content, 'Tool call result shown directly to user, no need for further actions')
        self.assertEqual(result[5].tool_call_id, 'call_456')
        self.assertEqual(result[5].name, 'direct_tool')

        # Direct AIMessage with actual tool result
        self.assertIsInstance(result[6], AIMessage)
        self.assertEqual(result[6].content, ['Direct result: test input'])

        # Verify all expected calls were made (should be only ONE LLM call)
        fake_llm.verify_all_called()

    def test_direct_and_standard_tool_calls(self):
        """Test worker with both direct and standard tool calls returns error for direct tool."""
        # Setup: Create a standard tool
        class StandardTool(BaseTool):
            name: str = "standard_tool"
            description: str = "A standard test tool"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Standard: {input_str}"

        # Setup: Create a direct tool
        class DirectTool(BaseTool):
            name: str = "direct_tool"
            description: str = "A tool that returns directly"
            return_direct: bool = True

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Direct: {input_str}"

        # Create tools with metadata
        standard_tool = StandardTool()
        standard_tool.metadata = {'tool_definition': Mock(
            name='standard_tool',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        direct_tool = DirectTool()
        direct_tool.metadata = {'tool_definition': Mock(
            name='direct_tool',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Please use both tools")

        # First LLM response with both tool calls
        tool_calls_response = AIMessage(
            content="I'll use both tools",
            tool_calls=[
                {
                    'id': 'call_std_1',
                    'name': 'standard_tool',
                    'args': {'input_str': 'standard input'}
                },
                {
                    'id': 'call_dir_1',
                    'name': 'direct_tool',
                    'args': {'input_str': 'direct input'}
                }
            ]
        )

        # Second LLM response after getting error for direct tool
        final_response = AIMessage(content="I'll call the direct tool separately now")

        # Setup fake LLM with expectations - TWO calls expected
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_calls_response
        ).expect_invoke(
            input=[
                input_message,
                tool_calls_response,
                ToolMessage(content='Standard: standard input', tool_call_id='call_std_1', name='standard_tool'),
                ToolMessage(content='Tool error: direct_tool must be called separately without other tools. Please call it in a separate request.', tool_call_id='call_dir_1', name='direct_tool')
            ],
            output=final_response
        )

        # Create mock context with tools
        context = StubWorkersContext(fake_llm)
        context._tools = {
            'standard_tool': standard_tool,
            'direct_tool': direct_tool
        }

        llm_config = BaseLLMConfig(model_ref="default", tools=['standard_tool', 'direct_tool'])
        worker = Worker(llm_config, context)
        worker._tools = {
            'standard_tool': standard_tool,
            'direct_tool': direct_tool
        }

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify the sequence of messages
        from llm_workers.core.api import WorkerNotification

        # Expected order:
        # 0. thinking_start
        # 1. thinking_end
        # 2. AIMessage with both tool_calls
        # 3. tool_start (standard_tool)
        # 4. tool_end (standard_tool)
        # 5. ToolMessage (standard_tool result)
        # 6. ToolMessage (direct_tool ERROR - not executed)
        # 7. thinking_start (second iteration)
        # 8. thinking_end (second iteration)
        # 9. Final AIMessage from LLM

        self.assertEqual(len(result), 10)

        # First iteration - LLM returns both tool calls
        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 2)
        self.assertEqual(result[2].tool_calls[0]['name'], 'standard_tool')
        self.assertEqual(result[2].tool_calls[1]['name'], 'direct_tool')

        # Standard tool execution
        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'tool_start')

        self.assertIsInstance(result[4], WorkerNotification)
        self.assertEqual(result[4].type, 'tool_end')

        self.assertIsInstance(result[5], ToolMessage)
        self.assertEqual(result[5].content, 'Standard: standard input')
        self.assertEqual(result[5].tool_call_id, 'call_std_1')
        self.assertEqual(result[5].name, 'standard_tool')

        # Direct tool error (not executed, just error message)
        self.assertIsInstance(result[6], ToolMessage)
        self.assertIn('must be called separately', result[6].content)
        self.assertEqual(result[6].tool_call_id, 'call_dir_1')
        self.assertEqual(result[6].name, 'direct_tool')

        # Second iteration - LLM responds to the error
        self.assertIsInstance(result[7], WorkerNotification)
        self.assertEqual(result[7].type, 'thinking_start')

        self.assertIsInstance(result[8], WorkerNotification)
        self.assertEqual(result[8].type, 'thinking_end')

        self.assertIsInstance(result[9], AIMessage)
        self.assertEqual(result[9].content, "I'll call the direct tool separately now")

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_multiple_direct_tool_calls(self):
        """Test worker with multiple direct tool calls merges results into single AIMessage."""
        # Setup: Create two direct tools
        class DirectTool1(BaseTool):
            name: str = "direct_tool_1"
            description: str = "First direct tool"
            return_direct: bool = True

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Result 1: {input_str}"

        class DirectTool2(BaseTool):
            name: str = "direct_tool_2"
            description: str = "Second direct tool"
            return_direct: bool = True

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Result 2: {input_str}"

        # Create tools with metadata
        direct_tool_1 = DirectTool1()
        direct_tool_1.metadata = {'tool_definition': Mock(
            name='direct_tool_1',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        direct_tool_2 = DirectTool2()
        direct_tool_2.metadata = {'tool_definition': Mock(
            name='direct_tool_2',
            confidential=False,
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Please use both direct tools")

        # LLM response with both direct tool calls
        tool_calls_response = AIMessage(
            content="I'll use both direct tools",
            tool_calls=[
                {
                    'id': 'call_dir_1',
                    'name': 'direct_tool_1',
                    'args': {'input_str': 'input A'}
                },
                {
                    'id': 'call_dir_2',
                    'name': 'direct_tool_2',
                    'args': {'input_str': 'input B'}
                }
            ]
        )

        # Setup fake LLM with expectations - only ONE call expected
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_calls_response
        )

        # Create mock context with tools
        context = StubWorkersContext(fake_llm)
        context._tools = {
            'direct_tool_1': direct_tool_1,
            'direct_tool_2': direct_tool_2
        }

        llm_config = BaseLLMConfig(model_ref="default", tools=['direct_tool_1', 'direct_tool_2'])
        worker = Worker(llm_config, context)
        worker._tools = {
            'direct_tool_1': direct_tool_1,
            'direct_tool_2': direct_tool_2
        }

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify the sequence of messages
        from llm_workers.core.api import WorkerNotification

        # Expected order:
        # 0. thinking_start
        # 1. thinking_end
        # 2. AIMessage with both tool_calls
        # 3. tool_start (direct_tool_1)
        # 4. tool_end (direct_tool_1)
        # 5. ToolMessage (direct_tool_1 stub)
        # 6. tool_start (direct_tool_2)
        # 7. tool_end (direct_tool_2)
        # 8. ToolMessage (direct_tool_2 stub)
        # 9. AIMessage with merged content (list)

        self.assertEqual(len(result), 10)

        # First iteration - LLM returns both direct tool calls
        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 2)
        self.assertEqual(result[2].tool_calls[0]['name'], 'direct_tool_1')
        self.assertEqual(result[2].tool_calls[1]['name'], 'direct_tool_2')

        # First direct tool execution
        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'tool_start')

        self.assertIsInstance(result[4], WorkerNotification)
        self.assertEqual(result[4].type, 'tool_end')

        self.assertIsInstance(result[5], ToolMessage)
        self.assertEqual(result[5].content, 'Tool call result shown directly to user, no need for further actions')
        self.assertEqual(result[5].tool_call_id, 'call_dir_1')
        self.assertEqual(result[5].name, 'direct_tool_1')

        # Second direct tool execution
        self.assertIsInstance(result[6], WorkerNotification)
        self.assertEqual(result[6].type, 'tool_start')

        self.assertIsInstance(result[7], WorkerNotification)
        self.assertEqual(result[7].type, 'tool_end')

        self.assertIsInstance(result[8], ToolMessage)
        self.assertEqual(result[8].content, 'Tool call result shown directly to user, no need for further actions')
        self.assertEqual(result[8].tool_call_id, 'call_dir_2')
        self.assertEqual(result[8].name, 'direct_tool_2')

        # Merged AIMessage with list content
        self.assertIsInstance(result[9], AIMessage)
        self.assertIsInstance(result[9].content, list)
        self.assertEqual(len(result[9].content), 2)
        self.assertEqual(result[9].content[0], 'Result 1: input A')
        self.assertEqual(result[9].content[1], 'Result 2: input B')

        # Verify all expected calls were made (only ONE LLM call)
        fake_llm.verify_all_called()

    def test_user_confirmed_execution(self):
        pass

    def test_user_not_confirmed_execution(self):
        pass

    def test_confidential_tool(self):
        """Test that confidential tool results are marked and filtered when sent to LLM."""
        # Setup: Create a confidential tool (must be return_direct)
        class ConfidentialTool(BaseTool):
            name: str = "confidential_tool"
            description: str = "A confidential tool"
            return_direct: bool = True

            def _run(self, query: str, **kwargs) -> str:
                return f"Secret data: {query}"

        # Create tool with metadata marking it as confidential
        confidential_tool = ConfidentialTool()
        confidential_tool.metadata = {'tool_definition': Mock(
            name='confidential_tool',
            confidential=True,  # Mark as confidential
            require_confirmation=None,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Get secret data")

        # LLM response with confidential tool call
        tool_call_response = AIMessage(
            content="I'll get the secret data",
            tool_calls=[{
                'id': 'call_secret',
                'name': 'confidential_tool',
                'args': {'query': 'password'}
            }]
        )

        # Setup fake LLM with expectations - only ONE call expected
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        )

        # Create mock context with tool
        context = StubWorkersContext(fake_llm)
        context._tools = {'confidential_tool': confidential_tool}

        llm_config = BaseLLMConfig(model_ref="default", tools=['confidential_tool'])
        worker = Worker(llm_config, context)
        worker._tools = {'confidential_tool': confidential_tool}

        # Execute: Stream worker with non-streaming (stream=False)
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify the sequence of messages
        from llm_workers.core.api import WorkerNotification, CONFIDENTIAL

        # Expected order:
        # 0. thinking_start
        # 1. thinking_end
        # 2. AIMessage with tool_call
        # 3. tool_start notification
        # 4. tool_end notification
        # 5. ToolMessage (stub)
        # 6. AIMessage with confidential result (marked with CONFIDENTIAL=True)

        self.assertEqual(len(result), 7)

        # First iteration - LLM returns tool call
        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 1)
        self.assertEqual(result[2].tool_calls[0]['name'], 'confidential_tool')

        # Tool execution
        self.assertIsInstance(result[3], WorkerNotification)
        self.assertEqual(result[3].type, 'tool_start')

        self.assertIsInstance(result[4], WorkerNotification)
        self.assertEqual(result[4].type, 'tool_end')

        # Stub ToolMessage
        self.assertIsInstance(result[5], ToolMessage)
        self.assertEqual(result[5].content, 'Tool call result shown directly to user, no need for further actions')
        self.assertEqual(result[5].tool_call_id, 'call_secret')
        self.assertEqual(result[5].name, 'confidential_tool')

        # Confidential AIMessage with actual tool result
        self.assertIsInstance(result[6], AIMessage)
        self.assertEqual(result[6].content, ['Secret data: password'])
        # Verify it's marked as confidential
        self.assertTrue(getattr(result[6], CONFIDENTIAL, False))

        # Verify all expected calls were made
        fake_llm.verify_all_called()

    def test_kwargs_passed_to_on_tool_start(self):
        mock_handler = Mock()
        mock_handler.mock_add_spec(BaseCallbackHandler)
        mock_handler.ignore_agent = False
        callback_manager = CallbackManager(handlers=[mock_handler])

        # Create a simple tool
        class SimpleTool(BaseTool):
            name: str = "simple_tool"
            description: str = "A simple test tool"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Processed: {input_str}"

        # Create the tool and run it with callbacks and extra kwargs
        tool = SimpleTool()

        # Run the tool with callbacks and custom kwargs
        tool.run(
            "test input",
            callbacks=callback_manager,
            ui_hint="Calling test tool",
        )

        # Check that the callback was called with the expected kwargs
        mock_handler.on_tool_start.assert_called_once()
        _, kwargs = mock_handler.on_tool_start.call_args
        self.assertEqual(kwargs.get("ui_hint"), "Calling test tool")

    def test_confirmation_request_yielded(self):
        """Test that a ConfirmationRequest is yielded when a tool needs confirmation."""
        # Create a tool that requires confirmation
        class ConfirmationTool(BaseTool):
            name: str = "confirmation_tool"
            description: str = "A tool requiring confirmation"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Executed: {input_str}"

        tool = ConfirmationTool()
        tool.metadata = {'tool_definition': Mock(
            name='confirmation_tool',
            confidential=False,
            require_confirmation=True,  # Requires confirmation
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Use the tool")

        # LLM response with tool call
        tool_call_response = AIMessage(
            content='',
            tool_calls=[{
                'name': 'confirmation_tool',
                'args': {'input_str': 'test'},
                'id': 'call_123'
            }]
        )

        # Setup fake LLM
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        )

        # Create context and worker
        context = StubWorkersContext(fake_llm, tools={'confirmation_tool': tool})
        llm_config = BaseLLMConfig(model_ref="default", tools=['confirmation_tool'])
        worker = Worker(llm_config, context)

        # Execute: Stream worker
        result = []
        for chunk in worker.stream([input_message], stream=False):
            result.extend(chunk)

        # Verify: ConfirmationRequest is yielded
        from llm_workers.core.api import WorkerNotification, ConfirmationRequest

        # Expected: thinking_start, thinking_end, AIMessage, ConfirmationRequest
        self.assertEqual(len(result), 4)

        self.assertIsInstance(result[0], WorkerNotification)
        self.assertEqual(result[0].type, 'thinking_start')

        self.assertIsInstance(result[1], WorkerNotification)
        self.assertEqual(result[1].type, 'thinking_end')

        self.assertIsInstance(result[2], AIMessage)
        self.assertEqual(len(result[2].tool_calls), 1)

        # The key assertion: ConfirmationRequest is yielded
        self.assertIsInstance(result[3], ConfirmationRequest)
        self.assertEqual(len(result[3].tool_calls), 1)
        self.assertIn('call_123', result[3].tool_calls)
        self.assertIn('confirmation_tool', result[3].tool_calls['call_123'].action)

        fake_llm.verify_all_called()

    def test_confirmation_approved_executes_tool(self):
        """Test that resuming with approved=True executes the tool."""
        # Create a tool that requires confirmation
        class ConfirmationTool(BaseTool):
            name: str = "confirmation_tool"
            description: str = "A tool requiring confirmation"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Executed: {input_str}"

        tool = ConfirmationTool()
        tool.metadata = {'tool_definition': Mock(
            name='confirmation_tool',
            confidential=False,
            require_confirmation=True,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Use the tool")

        # LLM responses
        tool_call_response = AIMessage(
            content='',
            tool_calls=[{
                'name': 'confirmation_tool',
                'args': {'input_str': 'test'},
                'id': 'call_123'
            }]
        )

        final_response = AIMessage(content="Tool executed successfully")

        # Setup fake LLM
        fake_llm = MockInvokeLLM()
        # First call: returns tool call
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        )
        # Second call: after tool execution, returns final response
        fake_llm.expect_invoke(
            input=[
                input_message,
                tool_call_response,
                ToolMessage(content='Executed: test', tool_call_id='call_123', name='confirmation_tool')
            ],
            output=final_response
        )

        # Create context and worker
        context = StubWorkersContext(fake_llm, tools={'confirmation_tool': tool})
        llm_config = BaseLLMConfig(model_ref="default", tools=['confirmation_tool'])
        worker = Worker(llm_config, context)

        # First invocation: get ConfirmationRequest
        result1 = []
        for chunk in worker.stream([input_message], stream=False):
            result1.extend(chunk)

        # Find the ConfirmationRequest
        from llm_workers.core.api import ConfirmationRequest, ConfirmationResponse
        confirmation_request = None
        for item in result1:
            if isinstance(item, ConfirmationRequest):
                confirmation_request = item
                break

        self.assertIsNotNone(confirmation_request)

        # Create ConfirmationResponse approving the tool call
        confirmation_response = ConfirmationResponse(approved_tool_calls=['call_123'])

        # Second invocation: resume with approved confirmation
        result2 = []
        for chunk in worker.stream([input_message, tool_call_response, confirmation_response], stream=False):
            result2.extend(chunk)

        # Verify: tool was executed and we got ToolMessage and final AIMessage
        tool_message_found = False
        for item in result2:
            if isinstance(item, ToolMessage):
                self.assertIn('Executed: test', item.content)
                tool_message_found = True

        self.assertTrue(tool_message_found, "ToolMessage was not found in results")

        fake_llm.verify_all_called()

    def test_confirmation_rejected_cancels_tool(self):
        """Test that resuming with approved=False yields cancellation messages."""
        # Create a tool that requires confirmation
        class ConfirmationTool(BaseTool):
            name: str = "confirmation_tool"
            description: str = "A tool requiring confirmation"

            def _run(self, input_str: str, **kwargs) -> str:
                return f"Executed: {input_str}"

        tool = ConfirmationTool()
        tool.metadata = {'tool_definition': Mock(
            name='confirmation_tool',
            confidential=False,
            require_confirmation=True,
            ui_hint_template=None,
            ui_hint=None,
            ui_hint_args=None
        )}

        # Setup messages
        input_message = HumanMessage(content="Use the tool")

        # LLM response with tool call
        tool_call_response = AIMessage(
            content='',
            tool_calls=[{
                'name': 'confirmation_tool',
                'args': {'input_str': 'test'},
                'id': 'call_123'
            }]
        )

        # Setup fake LLM
        fake_llm = MockInvokeLLM()
        fake_llm.expect_invoke(
            input=[input_message],
            output=tool_call_response
        )

        # Create context and worker
        context = StubWorkersContext(fake_llm, tools={'confirmation_tool': tool})
        llm_config = BaseLLMConfig(model_ref="default", tools=['confirmation_tool'])
        worker = Worker(llm_config, context)

        # First invocation: get ConfirmationRequest
        result1 = []
        for chunk in worker.stream([input_message], stream=False):
            result1.extend(chunk)

        # Find the ConfirmationRequest
        from llm_workers.core.api import ConfirmationRequest, ConfirmationResponse
        confirmation_request = None
        for item in result1:
            if isinstance(item, ConfirmationRequest):
                confirmation_request = item
                break

        self.assertIsNotNone(confirmation_request)

        # Create ConfirmationResponse rejecting the tool call (empty list)
        confirmation_response = ConfirmationResponse(approved_tool_calls=[])

        # Second invocation: resume with rejected confirmation
        result2 = []
        for chunk in worker.stream([input_message, tool_call_response, confirmation_response], stream=False):
            result2.extend(chunk)

        # Verify: cancellation ToolMessage was yielded
        cancellation_found = False
        for item in result2:
            if isinstance(item, ToolMessage):
                self.assertIn('canceled by user', item.content)
                self.assertEqual(item.tool_call_id, 'call_123')
                self.assertEqual(item.name, 'confirmation_tool')
                cancellation_found = True

        self.assertTrue(cancellation_found, "Cancellation ToolMessage was not found")

        fake_llm.verify_all_called()
