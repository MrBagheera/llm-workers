from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List, Literal, TypeVar, Generic, Generator
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from llm_workers.config import WorkersConfig, ModelDefinition, UserConfig, Json, \
    ToolsDefinitionOrReference
from llm_workers.starlark import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker

# Flag for confidential messages (not shown to LLM)
CONFIDENTIAL: str = 'confidential'


class UserContext(ABC):

    @property
    @abstractmethod
    def environment(self) -> Dict[str, str]:
        """Get environment vars (only default and explicitly listed)."""
        pass

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
    def evaluation_context(self) -> EvaluationContext:
        pass

    @property
    @abstractmethod
    def shared_tools(self) -> Dict[str, BaseTool]:
        pass

    @abstractmethod
    def get_tool(self, tool_ref: str, extra_tools: Optional[Dict[str, BaseTool]] = None) -> BaseTool:
        pass

    @abstractmethod
    def get_tools(self, scope: str, tool_refs: List[ToolsDefinitionOrReference]) -> List[BaseTool]:
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

    def __str__(self):
        return f"WorkerNotification(type={self.type}, run_id={self.run_id} text={self.text})"


Output = TypeVar('Output')
class ExtendedRunnable(ABC, Generic[Output]):
    """Abstract base class for Runnable with extended properties."""

    @abstractmethod
    def yield_notifications_and_result(
            self,
            evaluation_context: EvaluationContext,
            token_tracker: CompositeTokenUsageTracker,
            config: Optional[RunnableConfig],
            **kwargs: Any
    ) -> Generator[WorkerNotification, None, Output]:
        """Run the tool, (optionally) yields notifications, then result."""
        pass


class ExtendedExecutionTool(BaseTool, ExtendedRunnable[Any], ABC):
    """Base class for tools that support streaming and/or internal state notifications."""

    @abstractmethod
    def default_evaluation_context(self) -> EvaluationContext:
        """Get the default evaluation context for the tool."""
        pass

    def _run(
        self,
        *args: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,  # this is tool input
    ) -> Any:
        from llm_workers.worker_utils import extract_tool_results
        return extract_tool_results(
            self.name,
            # for tools called via this method, we cannot pass context or token tracker - use defaults
            self.yield_notifications_and_result(
                self.default_evaluation_context(),
                token_tracker=CompositeTokenUsageTracker(),
                config=config,
                **{'input': kwargs}
            )
        )

    def run_with_notifications(self,
        input: dict[str, Any],
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        **kwargs
    ) -> Generator[WorkerNotification, None, Any]:
        return self.yield_notifications_and_result(evaluation_context, token_tracker, config, **({**kwargs, 'input': input}))
