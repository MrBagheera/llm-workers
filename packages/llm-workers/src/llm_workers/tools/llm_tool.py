import json
import json
import logging
import re
from typing import Dict, Any, List, Union, Optional, Generator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import PrivateAttr, BaseModel, Field, ConfigDict

from llm_workers.api import WorkersContext, WorkerNotification, ExtendedExecutionTool, ConfirmationRequest
from llm_workers.config import ToolLLMConfig
from llm_workers.expressions import EvaluationContext
from llm_workers.token_tracking import CompositeTokenUsageTracker
from llm_workers.utils import LazyFormatter
from llm_workers.worker import Worker

_logger = logging.getLogger(__name__)


def extract_json_blocks(text: str, extract_json: Union[bool, str]) -> str:
    """
    Extract JSON blocks from text based on the extract_json parameter.
    
    Args:
        text: The input text to extract JSON from
        extract_json: Filtering option - True/"first", "last", "all", or "none"
        
    Returns:
        Extracted JSON as string or original text if no JSON found
    """
    if extract_json is None or extract_json == "none" or extract_json is False:
        return text
    
    # Find all JSON blocks
    json_pattern = r'(?:^|\n)```json\s*\n(.*?)\n```(?:\n|$)'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        return text  # Fallback to full message if no JSON found
    
    # Extract non-empty matches
    json_blocks = []
    for match in matches:
        if isinstance(match, tuple):
            # Find the non-empty group from the tuple
            for group in match:
                if group.strip():
                    json_blocks.append(group.strip())
                    break
        else:
            json_blocks.append(match.strip())
    
    if not json_blocks:
        return text
    
    # Apply filtering
    if extract_json is True or extract_json == "first":
        return json_blocks[0]
    elif extract_json == "last":
        return json_blocks[-1]
    elif extract_json == "all":
        return json.dumps(json_blocks, ensure_ascii=False)
    
    return text


class LLMToolInput(BaseModel):
    """Input schema for LLM tool."""
    model_config = ConfigDict(extra="allow")
    prompt: str = Field(description="Text prompt to send to the LLM")
    system_message: Optional[str] = Field(default=None, description="Optional system message to prepend to the conversation")


class LLMTool(ExtendedExecutionTool):
    _agent: Worker = PrivateAttr()
    _config: ToolLLMConfig = PrivateAttr()

    def __init__(self, agent: Worker, config: ToolLLMConfig, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        self._config = config

    def default_evaluation_context(self) -> EvaluationContext:
        return self._agent.context.evaluation_context

    def _extract_result(self, result: List[BaseMessage]) -> Any:
        """Extract text result and capture token usage from LLM response."""
        if len(result) == 0:
            return ""

        # Extract text content
        if len(result) == 1:
            text = str(result[0].text)
        elif len(result) > 1:
            # return only AI message(s)
            text = "\n".join([message.text for message in result if isinstance(message, AIMessage)])
        else:
            text = ""

        # Apply JSON filtering if configured
        if self._config.extract_json and self._config.extract_json != "none" and self._config.extract_json is not False:
            _logger.debug("Extracting JSON from LLM output (mode=%s):\n%s", self._config.extract_json, LazyFormatter(text))
            json_text = extract_json_blocks(text, self._config.extract_json)
            try:
                return json.loads(json_text)
                # Keeping just in case: this is a hack, but until we fix templating input JSON will arrive to LLM as single-quoted
                # so it may also produce single-quoted JSON outputs
                # return ast.literal_eval(json_text.replace("true", "True").replace("false", "False"))
            except (json.JSONDecodeError, ValueError, SyntaxError):
                _logger.warning("Failed to parse JSON from LLM output, returning as plain text:\n%s", json_text, exc_info=True)
                return json_text
        else:
            return text

    def yield_notifications_and_result(
        self,
        evaluation_context: EvaluationContext,
        token_tracker: CompositeTokenUsageTracker,
        config: Optional[RunnableConfig],
        **kwargs: Any
    ) -> Generator[WorkerNotification, None, Any]:
        """
        Calls LLM with given prompt, returns LLM output.
        """
        input = LLMToolInput(**kwargs['input'])

        messages = []
        if input.system_message:
            messages.append(SystemMessage(input.system_message))
        messages.append(HumanMessage(input.prompt))

        result: List[BaseMessage] = list()
        for e in self._agent.stream_with_notifications(input=messages, config=config, stream=False,
               **{**input.model_extra, 'evaluation_context': evaluation_context, 'token_tracker': token_tracker}):
            if isinstance(e, WorkerNotification):
                yield e
            elif isinstance(e, ConfirmationRequest):
                # TODO handle confirmations
                raise RuntimeError("LLM tries to use tools that require user confirmation, which are not supported in LLM tool.")
            else:
                result.append(e)

        # Capture token usage from AI messages
        if token_tracker:
            model_name = self._config.model_ref
            for message in result:
                token_tracker.update_from_message(message, model_name)

        return self._extract_result(result)


def build_llm_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> LLMTool:
    config = ToolLLMConfig(**tool_config)
    agent = Worker(config, context, scope='build_llm_tool')

    return LLMTool(
        agent=agent,
        config=config,
        name='llm',
        description='Calls LLM with given prompt, returns LLM output.',
        args_schema=LLMToolInput
    )
