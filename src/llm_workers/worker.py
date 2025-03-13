import logging
import sys
from typing import Optional, Any, List

from langchain.globals import get_verbose
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import (
    ensure_config,
    get_callback_manager_for_config,
)

from llm_workers.api import WorkersContext
from llm_workers.config import BaseLLMConfig
from llm_workers.utils import LazyFormatter

logger = logging.getLogger(__name__)


class Worker(Runnable[List[BaseMessage], List[BaseMessage]]):

    def __init__(self, llm_config: BaseLLMConfig, context: WorkersContext):
        self._system_message: Optional[SystemMessage] = None
        if llm_config.system_message is not None:
            self._system_message = SystemMessage(llm_config.system_message)
        self._llm = context.get_llm(llm_config.model_ref)
        self._tools = {}
        tools = []
        for tool_name in llm_config.tool_refs:
            tool = context.get_tool(tool_name)
            self._tools[tool_name] = tool
            tools.append(tool)
        if len(tools) > 0:
            self._llm = self._llm.bind_tools(tools)
        self._direct_tools = set([tool.name for tool in tools if tool.return_direct])

    def invoke(self, input: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        if self._system_message is not None:
            input = [self._system_message] + input

        callback_manager: CallbackManager = get_callback_manager_for_config(ensure_config(config))

        output: List[BaseMessage] = []
        while True:
            last: Optional[BaseMessage] = None
            for chunk in self._llm.stream(input, config, **kwargs):
                if isinstance(chunk, BaseMessageChunk):
                    if last is None:
                        last = chunk
                    elif isinstance(last, BaseMessageChunk) and last.id == chunk.id:
                        last = last + chunk
                    else:
                        self._append_llm_message(callback_manager, output, last, "intermediate LLM message")
                        last = chunk
                else:
                    if last is not None:
                        self._append_llm_message(callback_manager, output, last, "intermediate LLM message")
                    last = chunk
            if last is None:
                return [] # no output from LLM

            if isinstance(last, AIMessage) and len(last.tool_calls) > 0:
                self._append_llm_message(callback_manager, output, last, "last LLM message with tool calls")
                results = self._handle_tool_calls(last.tool_calls, config, **kwargs)
                if isinstance(results[0], AIMessage):   # return_direct tool calls
                    # remove last response from LLM
                    output.pop()
                    # return tool calls as if LLM responded with them
                    for result in results:
                        self._append_llm_message(callback_manager, output, result, "direct tool message")
                    return output
                else:
                    # append calls and results to input to continue LLM conversation
                    input.append(last)
                    input.extend(results)
                    # it is recommended to include tool calls and results in
                    # chat history for possible further use in the conversation
                    output.extend(results)
                    # continue to call LLM again
            else:
                # no tool calls, return LLM response
                self._append_llm_message(callback_manager, output, last, "last LLM message")
                return output

    @staticmethod
    def _append_llm_message(callback_manager: CallbackManager, output: List[BaseMessage], message: BaseMessage, log_info: str):
        logger.debug("Got %s:\n%r", log_info, LazyFormatter(message))
        if get_verbose():
            print(message.pretty_repr(), file = sys.stderr)
        if not isinstance(message, BaseMessageChunk) and len(message.content) > 0: # chunks events are streamed automatically
            callback_manager.on_custom_event(
                name = "on_ai_message",
                data = message
            )
        output.append(message)


    def _handle_tool_calls(self, tool_calls: List[ToolCall], config: Optional[RunnableConfig], **kwargs: Any):
        results = []
        use_direct_results = False
        for tool_call in tool_calls:
            if tool_call['name'] in self._direct_tools:
                use_direct_results = True
                break
        for tool_call in tool_calls:
            tool = self._tools[tool_call['name']]
            args = tool_call['args']
            logger.info("Calling tool %s with args: %r", tool.name, args)
            try:
                tool_output = tool.invoke(args, config, **kwargs)
            except Exception as e:
                logger.warning("Failed to call tool %s", tool.name, exc_info=True)
                tool_output = f"Tool Error: {e}"
            if use_direct_results:
                if not tool.return_direct:
                    logger.warning("Returning results of %s tool call as direct, as it is mixed with other return_direct tool calls", tool.name)
                response = AIMessage(content = tool_output, tool_call_id = tool_call['id'])
            else:
                response = ToolMessage(content = tool_output, tool_call_id = tool_call['id'], name = tool.name)
            logger.debug("Tool call result:\n%r", LazyFormatter(response))
            if get_verbose():
                print(response.pretty_repr(), file = sys.stderr)
            results.append(response)
        return results