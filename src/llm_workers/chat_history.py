import logging
from typing import Annotated, Union, Any

import yaml
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, AIMessageChunk, ToolMessageChunk
from pydantic import BaseModel, Discriminator, Tag

logger = logging.getLogger(__name__)

def _message_discriminator(v: Any) -> str | None:
    if isinstance(v, BaseMessage):
        return v.type
    elif isinstance(v, dict):
        return v.get('type')
    return None

ChatHistoryMessage = Annotated[
    Union[
        Annotated[HumanMessage, Tag("human")],
        Annotated[AIMessage, Tag("ai")],
        Annotated[AIMessageChunk, Tag("AIMessageChunk")],
        Annotated[ToolMessage, Tag("tool")],
        Annotated[ToolMessageChunk, Tag("ToolMessageChunk")],
    ],
    Discriminator(_message_discriminator)
]


class ChatHistory(BaseModel):
    script_name: str
    messages: list[ChatHistoryMessage]

    def save_to_yaml(self, file_path: str) -> str:
        """Save chat history to a YAML file."""
        file_path = ChatHistory._normalize_session_filename(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        return file_path

    @classmethod
    def load_from_yaml(cls, file_path: str) -> 'ChatHistory':
        """Load chat history from a YAML file."""
        file_path = ChatHistory._normalize_session_filename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)

        return ChatHistory(**content)

    @staticmethod
    def _normalize_session_filename(filename: str) -> str:
        """Normalize session filename by appending .chat.yaml suffix if missing."""
        if not filename.endswith('.chat.yaml'):
            filename += '.chat.yaml'
        return filename
