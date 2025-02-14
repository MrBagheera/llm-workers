import logging
import sys
from typing import Optional, Any, List, Iterator, Dict, AsyncIterator

from langchain.globals import get_verbose
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input

from llm_workers.api import LLMWorkersContext
from llm_workers.config import WorkerConfig
from llm_workers.llm import build_tool_calling_llm

logger = logging.getLogger(__name__)

class LlmWorker(Runnable[str, List[BaseMessage]]):
    def __init__(self, context: LLMWorkersContext):
        self._context = context
        self._llm = build_tool_calling_llm(self.config.chat, self._context)

    @property
    def config(self) -> WorkerConfig:
        return self._context.config

    @staticmethod
    # noinspection PyShadowingBuiltins
    def _transform_input(input: str| List[BaseMessage]) -> Input:
        if isinstance(input, str):
            return {"messages": [HumanMessage(input)]}
        elif isinstance(input, list):
            return {"messages": input}
        else:
            raise ValueError(f"Input '{input}' not supported.")

    @staticmethod
    def _transform_invoke_output(output: Dict[str, Any]) -> List[BaseMessage]:
        messages = output["messages"]
        if get_verbose():
            for message in messages:
                print(message.pretty_repr(), file=sys.stderr)
        return messages

    @staticmethod
    def _transform_stream_output(output: Dict[str, Any]) -> List[BaseMessage]:
        message = output["messages"][-1]
        if get_verbose():
            print(message.pretty_repr(), file=sys.stderr)
        return [message]

    @staticmethod
    async def _transform_async_iterator(source: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[List[BaseMessage]]:
        async for item in source:
            yield LlmWorker._transform_stream_output(item)

    # noinspection PyShadowingBuiltins
    def invoke(self, input: str | List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        llm_output = self._llm.invoke(input=LlmWorker._transform_input(input), config=config, streamMode = "values", **kwargs)
        return LlmWorker._transform_invoke_output(llm_output)

    # noinspection PyShadowingBuiltins
    def stream(self, input: str | List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Optional[Any]) -> Iterator[
        List[BaseMessage]]:
        llm_output = self._llm.stream(input=LlmWorker._transform_input(input), config=config, stream_mode = "values", **kwargs)
        return map(LlmWorker._transform_stream_output, llm_output)

    # noinspection PyShadowingBuiltins
    async def ainvoke(self, input: str | List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        llm_output = await self._llm.ainvoke(input=LlmWorker._transform_input(input), config=config, streamMode = "values", **kwargs)
        return LlmWorker._transform_invoke_output(llm_output)

    # noinspection PyShadowingBuiltins
    async def astream(self, input: str | List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs: Optional[Any]) -> \
    AsyncIterator[List[BaseMessage]]:
        llm_output = self._llm.astream(input=LlmWorker._transform_input(input), config=config, stream_mode = "values", **kwargs)
        return LlmWorker._transform_async_iterator(llm_output)

