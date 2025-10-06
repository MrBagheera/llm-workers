import json
import logging
from typing import Optional, Any, List, Iterator

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import (
    ensure_config,
    get_callback_manager_for_config,
)
from langchain_core.tools import BaseTool

from llm_workers.api import WorkersContext, ConfirmationRequest, ConfirmationRequestParam, \
    ExtendedBaseTool, CONFIDENTIAL, WorkerNotification, WorkerException
from llm_workers.config import BaseLLMConfig, ToolDefinition, ToolReference
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import LazyFormatter, call_tool

logger = logging.getLogger(__name__)

llm_calls_logger = logging.getLogger("llm_workers.llm_calls")

In = List[BaseMessage | WorkerNotification]
Out = List[BaseMessage | WorkerNotification]

class Worker(Runnable[In, Out]):

    def __init__(self, llm_config: BaseLLMConfig, context: WorkersContext, top_level: bool = False):
        self._llm_config = llm_config
        self._context = context
        self._system_message: Optional[SystemMessage] = None
        if llm_config.system_message is not None:
            self._system_message = SystemMessage(llm_config.system_message)
        self._llm = context.get_llm(llm_config.model_ref)
        self._tools = {}
        tools = []
        tool_refs: Optional[List[ToolReference]] = llm_config.tools
        if tool_refs is None:
            if top_level:
                tool_refs = [tool_def.name for tool_def in context.config.tools if not tool_def.name.startswith("_")]
            else:
                tool_refs = []
        tool_ref: ToolReference
        for tool_ref in tool_refs:
            tool = context.get_tool(tool_ref)
            self._tools[tool.name] = tool
            tools.append(tool)
        if len(tools) > 0:
            self._llm = self._llm.bind_tools(tools)
        self._direct_tools = set([tool.name for tool in tools if tool.return_direct])

    @property
    def model_ref(self) -> str:
        """Get the current model reference."""
        return self._llm_config.model_ref

    @model_ref.setter
    def model_ref(self, model_ref: str) -> None:
        """Set the model reference and reinitialize the LLM."""
        if model_ref == self._llm_config.model_ref:
            return  # No change needed
        
        self._llm_config.model_ref = model_ref
        new_llm = self._context.get_llm(model_ref)
        
        # Re-bind tools if we have any
        if len(self._tools) > 0:
            self._llm = new_llm.bind_tools(list(self._tools.values()))
        else:
            self._llm = new_llm

    def invoke(self, input: In, config: Optional[RunnableConfig] = None, stream: bool = False, **kwargs: Any) -> Out:
        result = []
        for message in self.stream_with_notifications(input, config, stream, **kwargs):
            if isinstance(message, BaseMessage):
                result.append(message)
        return result

    def stream(self, input: In, config: Optional[RunnableConfig] = None, stream: bool = True, **kwargs: Optional[Any]) -> Iterator[Out]:
        for message in self.stream_with_notifications(input, config, stream, **kwargs):
            yield [ message ]

    def stream_with_notifications(self, input: In, config: Optional[RunnableConfig], stream: bool, **kwargs: Any) -> Iterator[BaseMessage | WorkerNotification]:
        # leaving only BaseMessage-s items in input
        input = [message for message in input if isinstance(message, BaseMessage)]
        # prepend system message
        if self._system_message is not None:
            input = [self._system_message] + input
        # filter out confidential messages
        self._filter_outgoing_messages(input, 0)

        callback_manager: CallbackManager = get_callback_manager_for_config(ensure_config(config))

        while True:
            yield WorkerNotification.thinking_start()
            llm_response: Optional[BaseMessage] = None
            for llm_message in self._invoke_llm(stream, input, config, **kwargs):
                if isinstance(llm_message, BaseMessage):
                    llm_response = llm_message
                    break
                else: # notification
                    yield llm_message
            yield WorkerNotification.thinking_end()
            if not llm_response:
                raise WorkerException(f"Invoking LLM resulted no message")
            self._log_llm_message(llm_response, "LLM message")
            yield llm_response # return LLM message (possibly with calls)

            if isinstance(llm_response, AIMessage) and len(llm_response.tool_calls) > 0:
                if self._check_if_user_cancels_execution(callback_manager, llm_response.tool_calls):
                    for tool_call in llm_response.tool_calls:
                        cancel_message = ToolMessage(
                            content = "Tool error: execution canceled by user",
                            tool_call_id = tool_call['id'],
                            name = tool_call['name']
                        )
                        self._log_llm_message(cancel_message, "canceled tool call")
                        yield cancel_message # return canceled tool call
                    return

                tool_results = []
                for tool_result in self._handle_tool_calls(llm_response.tool_calls, config, kwargs):
                    if isinstance(tool_result, ToolMessage):
                        tool_results.append(tool_result)
                        yield tool_result
                    elif isinstance(tool_result, AIMessage):
                        yield tool_result
                        return
                    else:
                        yield tool_result # return WorkerNotification-s immediately

                # Continue LLM conversation with tool results
                input.append(llm_response)
                input.extend(tool_results)
            else:
                # no tool calls
                return

    @staticmethod
    def _filter_outgoing_messages(input, next_index):
        for i in range(next_index, len(input)):
            message = input[i]
            if isinstance(message, AIMessage):
                # filter confidential messages
                if getattr(message, CONFIDENTIAL, False):
                    message = message.model_copy(update={'content': '[CONFIDENTIAL]'}, deep=False)
                    input[i] = message

    def _invoke_llm(self, stream: bool, input: List[BaseMessage], config: Optional[RunnableConfig], **kwargs: Any) -> Iterator[BaseMessage | WorkerNotification]:
        if llm_calls_logger.isEnabledFor(logging.DEBUG):
            llm_calls_logger.debug("Calling LLM with input:\n%r", LazyFormatter(input))
        if stream:
            # reassembling message from chunks
            last: Optional[BaseMessage] = None
            for message in self._llm.stream(input, config, **kwargs):
                if last is None:
                    last = message
                else:
                    last += message
                yield from self.extract_notifications(message_id=last.id, index=0, content=message.content)
            yield last
        else:
            yield self._llm.invoke(input, config)

    @staticmethod
    def extract_notifications(message_id: Optional[str], index: int, content: any) -> Iterator[WorkerNotification]:
        if isinstance(content, str):
            yield WorkerNotification.ai_output_chunk(message_id=message_id, index=index, text=content)
        elif isinstance(content, list):
            index = 0
            for block in content:
                yield from Worker.extract_notifications(message_id=message_id, index=index, content=block)
                index += 1
        elif isinstance(content, dict):
            # noinspection PyShadowingBuiltins
            type = content.get('type', None)
            index = content.get('index', index)
            if type == 'reasoning_content':
                reasoning_content = content.get("reasoning_content", {})
                text = reasoning_content.get("text", None)
                if text:
                    yield WorkerNotification.ai_reasoning_chunk(message_id=message_id, index=index, text=str(text))
            elif type == 'reasoning':
                if 'summary' in content: # OpenAI GPT-5
                    content = content.get("summary")
                    if isinstance(content, list) and len(content) > 0:
                        content = content[0]
                if isinstance(content, dict):
                    text = content.get("text", None)
                    if text:
                        yield WorkerNotification.ai_reasoning_chunk(message_id=message_id, index=index, text=str(text))
            else:
                text = content.get("text", None)
                if text:
                    yield WorkerNotification.ai_output_chunk(message_id=message_id, index=index, text=str(text))


    @staticmethod
    def _log_llm_message(message: BaseMessage, log_info: str):
        logger.debug("Got %s:\n%r", log_info, LazyFormatter(message, trim=False))

    def _use_direct_results(self, tool_calls: List[ToolCall]):
        """Check if any of the tool calls are direct_result tools."""
        for tool_call in tool_calls:
            if tool_call['name'] in self._direct_tools:
                return True
        return False

    def _handle_tool_calls(self, tool_calls: List[ToolCall], config: Optional[RunnableConfig], kwargs: dict[str, Any]) -> Iterator[BaseMessage | WorkerNotification]:
        # direct tools fail if there is any non-direct tool call
        direct_tools_fail = any(tc['name'] not in self._direct_tools for tc in tool_calls)

        direct_tools_results = []
        has_confidential_results = False
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            if tool_name not in self._tools:
                logger.warning("Failed to call tool %s: no such tool", tool_name, exc_info=True)
                content = f"Tool error: no such tool %s" % tool_name
                response = ToolMessage(content = content, tool_call_id = tool_call['id'], name = tool_name)
                self._log_llm_message(response, "tool call message")
                yield response
                continue
            tool: BaseTool = self._tools[tool_name]
            tool_definition: ToolDefinition = tool.metadata['tool_definition']
            args: dict[str, Any] = tool_call['args']
            logger.info("Calling tool %s with args:\n%r", tool.name, LazyFormatter(args))

            if tool.return_direct and direct_tools_fail:
                content = f"Tool error: {tool.name} must be called separately without other tools. Please call it in a separate request."
                response = ToolMessage(content = content, tool_call_id = tool_call['id'], name = tool.name)
                self._log_llm_message(response, "direct tool error")
                yield response
                continue

            tool_output: Any = None
            token_tracker = CompositeTokenUsageTracker()
            for e in call_tool(tool, args, token_tracker, config, kwargs):
                if isinstance(e, WorkerNotification):
                    yield e
                else:
                    tool_output = e

            tool_message: ToolMessage
            content: str
            if isinstance(tool_output, ToolMessage):
                tool_message = tool_output
                tool_message.tool_call_id = tool_call['id']
                tool_message.name = tool.name
                content = tool_message.content
            else:
                content = tool_output if isinstance(tool_output, str) else json.dumps(tool_output)
                tool_message = ToolMessage(content = content, tool_call_id = tool_call['id'], name = tool.name)
            if not token_tracker.is_empty:
                token_tracker.attach_usage_to_message(tool_message)

            if tool.return_direct:
                tool_message = ToolMessage(
                    content = "Tool call result shown directly to user, no need for further actions",
                    tool_call_id = tool_call['id'],
                    name = tool.name
                )
                yield tool_message
                direct_tools_results.append(content.strip())
                if self._is_confidential(tool, tool_definition):
                    has_confidential_results = True
            else:
                self._log_llm_message(tool_message, "tool call message")
                yield tool_message

        # merge all direct tools results into single AIMessage
        if len(direct_tools_results) == 0:
            return

        direct_response = AIMessage(content = direct_tools_results)
        if has_confidential_results:
            direct_response = direct_response.model_copy(update={CONFIDENTIAL: True}, deep=False)
        self._log_llm_message(direct_response, "direct tool message")
        yield direct_response

    @staticmethod
    def _is_confidential(tool: BaseTool, tool_definition: ToolDefinition) -> bool:
        if tool_definition.confidential is not None:
            return tool_definition.confidential
        elif isinstance(tool, ExtendedBaseTool):
            return tool.confidential
        return False

    def _check_if_user_cancels_execution(self, callback_manager: CallbackManager, tool_calls: list[ToolCall]) -> bool:
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            if tool_name not in self._tools:
                continue
            tool: BaseTool = self._tools[tool_name]
            tool_definition: ToolDefinition = tool.metadata.get('tool_definition')
            args: dict[str, Any] = tool_call['args']

            if tool_definition.require_confirmation is not None:
                if not tool_definition.require_confirmation:
                    continue
            elif isinstance(tool, ExtendedBaseTool):
                if not tool.needs_confirmation(args):
                    continue
            else:
                continue

            request: Optional[ConfirmationRequest] = None
            if isinstance(tool, ExtendedBaseTool):
                request = tool.make_confirmation_request(args)
            if request is None:
                request = ConfirmationRequest(
                    action = f"run the tool {tool.name} with following input",
                    params = [ConfirmationRequestParam(name=key, value=value) for key, value in args.items()]
                )

            callback_manager.on_custom_event(
                name = 'request_confirmation',
                data = request
            )

            if not request.approved:
                return True

        return False