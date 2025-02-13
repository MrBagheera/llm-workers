
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMWorkersContext(ABC):
    @abstractmethod
    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None ):
        pass

    @abstractmethod
    def get_llm(self, llm_name: str):
        pass