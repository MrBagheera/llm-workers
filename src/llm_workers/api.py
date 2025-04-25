
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from llm_workers.config import WorkersConfig, ToolDefinition

# Flag for confidential messages (not shown to LLM)
CONFIDENTIAL: str = 'confidential'



class WorkersContext(ABC):

    @property
    @abstractmethod
    def config(self) -> WorkersConfig:
        pass

    @abstractmethod
    def get_tool(self, tool_name: str) -> BaseTool:
        pass

    @abstractmethod
    def get_tool_definition(self, tool_name: str) -> ToolDefinition:
        pass

    @abstractmethod
    def get_llm(self, llm_name: str) -> BaseChatModel:
        pass

    @abstractmethod
    def get_start_tool_message(self, tool_name: str, inputs: dict[str, any]) -> str:
        pass


ToolFactory = Callable[[WorkersContext, Dict[str, Any]], BaseTool]


class WorkerException(Exception):
    """Custom exception for worker-related errors."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.message = message
        self.__cause__ = cause  # Pass the cause of the exception

    def __str__(self):
        return self.message


class ConfirmationRequestParam:
    """Class representing a parameter for a confirmation request."""
    name: str
    value: Any
    format: Optional[str] = None

    def __init__(self, name: str, value: Any, format: Optional[str] = None):
        self.name = name
        self.value = value
        self.format = format

class ConfirmationRequest:
    """Class representing a confirmation request."""
    action: str
    params: List[ConfirmationRequestParam]
    approved: bool = False
    reject_reason: Optional[str] = None

    def __init__(self, action: str, params: List[ConfirmationRequestParam]):
        self.action = action
        self.args = params


class ExtendedBaseTool(ABC):
    """Abstract base class for tools with extended properties."""

    confidential: bool = False

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        """Check if the tool requires confirmation for the given input."""
        return False

    def make_confirmation_request(self, input: dict[str, Any]) -> Optional[ConfirmationRequest]:
        """Create a custom confirmation request based on the input."""
        return None

    @abstractmethod
    def get_ui_hint(self, input: dict[str, Any]) -> str:
        pass

