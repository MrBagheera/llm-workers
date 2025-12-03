
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List, Literal, Iterable, TypeVar, Generic
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_workers.config import WorkersConfig, ToolReference, ModelDefinition, UserConfig, Json
from llm_workers.token_tracking import CompositeTokenUsageTracker

# Flag for confidential messages (not shown to LLM)
CONFIDENTIAL: str = 'confidential'



class UserContext(ABC):

    @property
    @abstractmethod
    def user_config(self) -> UserConfig:
        """Get the user configuration."""
        pass

    @property
    @abstractmethod
    def models(self) -> List[ModelDefinition]:
        """Get list of available model definitions."""
        pass

    @abstractmethod
    def get_llm(self, llm_name: str) -> BaseChatModel:
        pass


class WorkersContext(ABC):

    @property
    @abstractmethod
    def config(self) -> WorkersConfig:
        pass

    @property
    @abstractmethod
    def get_public_tools(self) -> List[BaseTool]:
        pass

    @abstractmethod
    def get_tool(self, tool_ref: ToolReference) -> BaseTool:
        pass

    @abstractmethod
    def get_llm(self, llm_name: str) -> BaseChatModel:
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


class ConfirmationRequestParam(BaseModel):
    """Class representing a parameter for a confirmation request."""
    name: str
    value: Json
    format: Optional[str] = None

class ConfirmationRequestToolCallDescription(BaseModel):
    """Class representing a confirmation request."""
    action: str
    params: List[ConfirmationRequestParam]

class ConfirmationRequest(BaseModel):
    """Class representing a confirmation request from agent to UI."""
    # tools calls from last AIMessage that need confirmation, mapped by id
    tool_calls: dict[str, ConfirmationRequestToolCallDescription]

class ConfirmationResponse(BaseModel):
    """Class representing a confirmation response from UI to agent."""
    # confirmation for tools calls from last AIMessage
    # calls that required confirmation but are not in this list are considered rejected
    approved_tool_calls: List[str]


class ExtendedBaseTool(ABC):
    """Abstract base class for tools with extended properties."""

    confidential: bool = False

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        """Check if the tool requires confirmation for the given input."""
        return False

    def make_confirmation_request(self, input: dict[str, Any]) -> Optional[ConfirmationRequestToolCallDescription]:
        """Create a custom confirmation request based on the input."""
        return None

    @abstractmethod
    def get_ui_hint(self, input: dict[str, Any]) -> str:
        pass


Input = TypeVar('Input')
Output = TypeVar('Output')


class ExtendedRunnable(ABC, Generic[Input, Output]):
    """Abstract base class for Runnable with extended properties."""

    @abstractmethod
    def _stream(
            self,
            token_tracker: Optional[CompositeTokenUsageTracker],
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Iterable[Any]:
        """Internal method to run the tool and optionally yield notifications."""
        pass


class ExtendedExecutionTool(BaseTool, ExtendedRunnable[dict[str, Any], Any], ABC):
    """Base class for tools that support streaming and/or internal state notifications."""

    def _run(
            self,
            *args: Any,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Any:
        for chunk in self._stream(token_tracker=None, config = config, **kwargs):
            if isinstance(chunk, WorkerNotification):
                continue
            return chunk
        return None

    def stream_with_notifications(self, input: dict[str, Any], token_tracker: CompositeTokenUsageTracker, config: Optional[RunnableConfig]):
        return self._stream(token_tracker = token_tracker, config = config, **input)


class WorkerNotification:
    """Notifications about worker state changes."""
    type: Literal['thinking_start', 'thinking_end', 'tool_start', 'tool_end', 'ai_output_chunk', 'ai_reasoning_chunk']
    message_id: Optional[str] = None
    index: int = 0
    text: Optional[str] = None
    run_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None

    @staticmethod
    def thinking_start() -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='thinking_start'
        return n

    @staticmethod
    def thinking_end() -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='thinking_end'
        return n

    @staticmethod
    def tool_start(text: str, run_id: UUID, parent_run_id: Optional[UUID] = None) -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='tool_start'
        n.text=text
        n.run_id=run_id
        n.parent_run_id=parent_run_id
        return n

    @staticmethod
    def tool_end(run_id: UUID) -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='tool_end'
        n.run_id=run_id
        return n

    @staticmethod
    def ai_output_chunk(message_id: Optional[str], index: int, text: str) -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='ai_output_chunk'
        n.message_id = message_id
        n.index=index
        n.text=text
        return n

    @staticmethod
    def ai_reasoning_chunk(message_id: Optional[str], index: int, text: str) -> 'WorkerNotification':
        n = WorkerNotification()
        n.type='ai_reasoning_chunk'
        n.message_id = message_id
        n.index=index
        n.text=text
        return n