from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, Field, Tag, Discriminator

from llm_workers.llm import BaseLLMConfig
from llm_workers.tools.custom_tools_base import Json
from llm_workers.tools.llm_tool import LLMToolDefinition
from llm_workers.tools.stub_tool import StubToolDefinition
from llm_workers.tools.t2_ai_wrapper import T2AiWrapperToolDefinition
from llm_workers.tools.tool_binding import ToolBindingDefinition


def custom_tool_discriminator_value(v: Any) -> str:
    if isinstance(v, dict):
        result = v.get('type')
    else:
        result = getattr(v, 'type')
    return 'tool_binding' if result is None else result

CustomToolDefinition = Annotated[
    Annotated[ToolBindingDefinition, Tag('tool_binding')] |
    Annotated[T2AiWrapperToolDefinition, Tag('t2-ai-wrapper')] |
    Annotated[StubToolDefinition, Tag('stub')] |
    Annotated[LLMToolDefinition, Tag('LLM')],
    Field(discriminator=Discriminator(custom_tool_discriminator_value))
]

class ModelConfig(BaseModel):
    name: str
    provider: str
    model: str
    model_params: Json = None

class MainConfig(BaseLLMConfig):
    default_prompt: str | None = None

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
    ) -> Self:
        obj = super().model_validate(obj, strict=strict, from_attributes=from_attributes, context=context)
        # either system_prompt or prompt must be set
        if obj.system_message and obj.prompt:
            raise ValueError("system_prompt and prompt cannot be set at the same time")
        return obj

class WorkerConfig(BaseModel):
    models: list[ModelConfig]
    custom_tools: list[CustomToolDefinition] = ()
    main: MainConfig

def load_config(file_path: str) -> WorkerConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkerConfig(**config_data)
