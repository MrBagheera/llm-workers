
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from llm_workers.config import WorkersConfig


class WorkersContext(ABC):

    @property
    @abstractmethod
    def config(self) -> WorkersConfig:
        pass

    @abstractmethod
    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None) -> BaseTool:
        pass

    @abstractmethod
    def get_llm(self, llm_name: str) -> BaseChatModel:
        pass


ToolFactory = Callable[[WorkersContext, Dict[str, Any]], BaseTool]

