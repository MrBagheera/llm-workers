from typing import List, Union, Annotated

import yaml
from pydantic import BaseModel, Field

from llm_workers.tools.debug_tool import DebugToolConfig
from llm_workers.tools.llm_tool import LLMToolConfig

CustomTool = Annotated[
    Union[DebugToolConfig, LLMToolConfig],
    Field(discriminator='tool_type')
]

class Main(BaseModel):
    provider: str
    model: str
    instructions: str
    interactive: bool
    tool_refs: List[str]
    verbose: bool = False
    debug: bool = False

class WorkerConfig(BaseModel):
    custom_tools: list[CustomTool]
    main: Main

def load_config(file_path: str) -> WorkerConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkerConfig(**config_data)
