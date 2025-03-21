import logging
import sys
from typing import Optional, Any, List, Iterator

from langchain.globals import get_verbose
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import (
    ensure_config,
    get_callback_manager_for_config,
)
from langchain_core.tools import BaseTool

from llm_workers.api import WorkersContext, ConfirmationRequest, ConfirmationRequestParam, \
    ToolExecutionRejectedException, ExtendedBaseTool, CONFIDENTIAL
from llm_workers.config import BaseLLMConfig, ToolDefinition
from llm_workers.utils import LazyFormatter

logger = logging.getLogger(__name__)


class Worker(Runnable[List[BaseMessage], List[BaseMessage]]):

    def __init__(self, llm_config: BaseLLMConfig, context: WorkersContext, top_level: bool = False):
        self._llm_config = llm_config
        self._system_message: Optional[SystemMessage] = None
        if llm_config.system_message is not None:
            self._system_message = SystemMessage(llm_config.system_message)
        self._llm = context.get_llm(llm_config.model_ref)
        self._tools = {}
        self._tool_definitions = {}
        tools = []
        tool_refs = llm_config.tool_refs
        if tool_refs is None:
            if top_level:
                tool_refs = [tool_def.name for tool_def in context.config.tools if not tool_def.name.startswith("_")]
            else:
                tool_refs = []
        for tool_name in tool_refs:
            tool = context.get_tool(tool_name)
            self._tools[tool_name] = tool
            self._tool_definitions[tool_name] = context.get_tool_definition(tool_name)
            tools.append(tool)
        if len(tools) > 0:
            self._llm = self._llm.bind_tools(tools)
        self._direct_tools = set([tool.name for tool in tools if tool.return_direct])

    def invoke(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, stream: bool = False, **kwargs: Any) -> List[BaseMessage]:
        result = []
        for message in self._stream(input, config, stream, **kwargs):
            result.append(message)
        return result

    def stream(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, stream: bool = False, **kwargs: Optional[Any]) -> Iterator[List[BaseMessage]]:
        for message in self._stream(input, config, stream, **kwargs):
            yield [ message ]

    def _stream(self, input: List[BaseMessage], config: Optional[RunnableConfig], stream: bool, **kwargs: Any) -> Iterator[BaseMessage]:

        if self._system_message is not None:
            input = [self._system_message] + input
        else:
            input = input.copy()
        self._filter_outgoing_messages(input, 0)

        callback_manager: CallbackManager = get_callback_manager_for_config(ensure_config(config))

        while True:
            response = self._invoke_llm(stream, input, config, **kwargs)

            if isinstance(response, AIMessage) and len(response.tool_calls) > 0:
                self._log_llm_message(response, "LLM message with tool calls")
                if self._use_direct_results(response.tool_calls):
                    # include LLM message without tool calls, if not empty
                    text = response.text()
                    if len(text) > 0:
                        yield AIMessage(content=text)
                    # return tool call results as if LLM responded with them
                    results = self._handle_tool_calls(callback_manager, response.tool_calls, True, config, **kwargs)
                    for result in results:
                        yield result
                    return
                else:
                    yield response
                    results = self._handle_tool_calls(callback_manager, response.tool_calls, False, config, **kwargs)
                    # it is recommended to include tool calls and results in
                    # chat history for possible further use in the conversation
                    for result in results:
                        yield result
                    # append calls and results to input to continue LLM conversation
                    input.append(response)
                    input.extend(results)
                    # continue to call LLM again
            else:
                # no tool calls, return LLM response
                self._log_llm_message(response, "LLM message")
                yield response
                return

    def _filter_outgoing_messages(self, input, next_index):
        for i in range(next_index, len(input)):
            message = input[i]
            if isinstance(message, AIMessage):
                # filter confidential messages
                if getattr(message, CONFIDENTIAL, False):
                    message = message.model_copy(update={'content': '[CONFIDENTIAL]'}, deep=False)
                    input[i] = message
                if self._llm_config.remove_past_reasoning:
                    # do not send reasoning content
                    if isinstance(message.content, list):
                        fixed_content: Optional[list] = None
                        for block in message.content:
                            if isinstance(block, dict):
                                if block.get("type") == "reasoning_content":
                                    if fixed_content is None:
                                        fixed_content = message.content.copy()
                                    fixed_content.remove(block)
                        if fixed_content is not None:
                            message = message.model_copy(update={'content': fixed_content}, deep=False)
                            input[i] = message

    def _invoke_llm(self, stream: bool, input: List[BaseMessage], config: Optional[RunnableConfig], **kwargs: Any) -> BaseMessage:
        # converse-bedrock doesn't support "stream" attribute, have to work around it
        if stream:
            # reassembling message from chunks
            last: Optional[BaseMessage] = None
            for message in self._llm.stream(input, config, **kwargs):
                if last is None:
                    last = message
                else:
                    last += message
            return last
        else:
            return self._llm.invoke(input, config, **{**kwargs, "stream": stream})

    @staticmethod
    def _log_llm_message(message: BaseMessage, log_info: str):
        logger.debug("Got %s:\n%r", log_info, LazyFormatter(message))
        if get_verbose():
            print(message.pretty_repr(), file = sys.stderr)

    def _use_direct_results(self, tool_calls: List[ToolCall]):
        """Check if any of the tool calls are direct_result tools."""
        for tool_call in tool_calls:
            if tool_call['name'] in self._direct_tools:
                return True
        return False

    def _handle_tool_calls(self, callback_manager: CallbackManager, tool_calls: List[ToolCall], use_direct_results: bool, config: Optional[RunnableConfig], **kwargs: Any):
        results = []
        for tool_call in tool_calls:
            tool: BaseTool = self._tools[tool_call['name']]
            tool_definition: ToolDefinition = self._tool_definitions[tool_call['name']]
            args: dict[str, Any] = tool_call['args']
            logger.info("Calling tool %s with args: %r", tool.name, args)
            execution_rejected = False
            try:
                self._request_confirmation_if_needed(callback_manager, tool, tool_definition, args)
                tool_output = tool.invoke(args, config, **kwargs)
            except ToolExecutionRejectedException as e:
                logger.info("Execution of tool %s rejected, reason: %s", tool.name, e.reason)
                if e.reason is not None:
                    tool_output = f"Tool Error: execution rejected by user, reason: {e.reason}"
                else:
                    tool_output = f"Tool Error: execution rejected, reason: cannot ask confirmation in non-interactive environment"
                execution_rejected = True
            except Exception as e:
                logger.warning("Failed to call tool %s", tool.name, exc_info=True)
                tool_output = f"Tool Error: {e}"
            if use_direct_results:
                if not tool.return_direct:
                    logger.warning("Returning results of %s tool call as direct, as it is mixed with other return_direct tool calls", tool.name)
                response = AIMessage(content = tool_output, tool_call_id = tool_call['id'])
                if not execution_rejected and self._is_confidential(tool, tool_definition):
                    response = response.model_copy(update={CONFIDENTIAL: '[CONFIDENTIAL]'}, deep=False)
                self._log_llm_message(response, "direct tool message")
            else:
                response = ToolMessage(content = tool_output, tool_call_id = tool_call['id'], name = tool.name)
                self._log_llm_message(response, "tool call message")
            results.append(response)
        return results

    @staticmethod
    def _is_confidential(tool: BaseTool, tool_definition: ToolDefinition) -> bool:
        if tool_definition.confidential is not None:
            return tool_definition.confidential
        elif isinstance(tool, ExtendedBaseTool):
            return tool.confidential
        return False

    @staticmethod
    def _request_confirmation_if_needed(callback_manager: CallbackManager, tool: BaseTool, tool_definition: ToolDefinition, args: dict[str, Any]):
        if tool_definition.require_confirmation is not None:
            if not tool_definition.require_confirmation:
                return
        elif isinstance(tool, ExtendedBaseTool):
            if not tool.needs_confirmation(args):
                return
        else:
            return

        # TODO support confirmation_prompt and confirmation_params in tool config
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
            raise ToolExecutionRejectedException(request.reject_reason)
