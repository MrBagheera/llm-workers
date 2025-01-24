from typing import List, Union, Annotated, Any, Self

import yaml
from pydantic import BaseModel, Field

from llm_workers.llm import BaseLLMConfig
from llm_workers.tools.debug_tool import DebugToolConfig
from llm_workers.tools.llm_tool import LLMToolConfig

CustomToolConfig = Annotated[
    Union[DebugToolConfig, LLMToolConfig],
    Field(discriminator='tool_type')
]

class MainConfig(BaseLLMConfig):
    default_prompt: str | None = None
    verbose: bool = False
    debug: bool = False

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
    custom_tools: list[CustomToolConfig]
    main: MainConfig

def load_config(file_path: str) -> WorkerConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkerConfig(**config_data)
