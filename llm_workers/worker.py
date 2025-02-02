import sys
from typing import Optional, Any, List, Iterator, Dict, AsyncIterator

from langchain.chat_models import init_chat_model
from langchain.globals import get_verbose
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input

from llm_workers.config import load_config, WorkerConfig
from llm_workers.llm import build_tool_calling_llm
from llm_workers.tools.registry import ToolRegistry


class LlmWorker(Runnable[str, List[BaseMessage]]):
    def __init__(self, config_filename: str):
        self._config = load_config(config_filename)
        self._model_registry = dict[str, BaseChatModel]()
        for model in self._config.models:
            self._model_registry[model.name] = init_chat_model(model.model, model_provider=model.provider, configurable_fields=None, temperature=0)
        self._tool_registry = ToolRegistry()
        self._tool_registry.register_custom_tools(self._model_lookup, self._config.custom_tools)
        self._llm = build_tool_calling_llm(self._config.main, models_lookup = self._model_lookup, tools_lookup=self._tool_registry.resolve_tool_refs)

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def default_prompt(self) -> str:
        return self._config.main.default_prompt

    def _model_lookup(self, name: str) -> BaseChatModel:
        model = self._model_registry[name]
        if model is None:
            raise ValueError(f"Model '{name}' not found.")
        return model

    @staticmethod
    # noinspection PyShadowingBuiltins
    def _transform_input(input: str) -> Input:
        return {"messages": [HumanMessage(input)]}

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
            message.pretty_print()
        return [message]

    @staticmethod
    async def _transform_async_iterator(source: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[List[BaseMessage]]:
        async for item in source:
            yield LlmWorker._transform_stream_output(item)

    # noinspection PyShadowingBuiltins
    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        llm_output = self._llm.invoke(input=LlmWorker._transform_input(input), config=config, streamMode = "values", **kwargs)
        return LlmWorker._transform_invoke_output(llm_output)

    # noinspection PyShadowingBuiltins
    def stream(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Optional[Any]) -> Iterator[
        List[BaseMessage]]:
        llm_output = self._llm.stream(input=LlmWorker._transform_input(input), config=config, stream_mode = "values", **kwargs)
        return map(LlmWorker._transform_stream_output, llm_output)

    # noinspection PyShadowingBuiltins
    async def ainvoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        llm_output = await self._llm.ainvoke(input=LlmWorker._transform_input(input), config=config, streamMode = "values", **kwargs)
        return LlmWorker._transform_invoke_output(llm_output)

    # noinspection PyShadowingBuiltins
    async def astream(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Optional[Any]) -> \
    AsyncIterator[List[BaseMessage]]:
        llm_output = self._llm.astream(input=LlmWorker._transform_input(input), config=config, stream_mode = "values", **kwargs)
        return LlmWorker._transform_async_iterator(llm_output)

