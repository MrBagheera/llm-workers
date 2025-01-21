from typing import List, Union, Literal, Annotated

from pydantic import BaseModel, Field


class ToolParam(BaseModel):
    description: str
    type: str
    optional: bool

class CustomToolBase(BaseModel):
    description: str
    params: dict[str, ToolParam]

class DebugTool(CustomToolBase):
    tool_type: Literal['debug']
    result_text: str

class LLMTool(CustomToolBase):
    tool_type: Literal['LLM']
    tool_refs: List[str] = ()
    instructions: str

CustomTool = Annotated[
    Union[DebugTool, LLMTool],
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
    custom_tools: dict[str, CustomTool]
    main: Main

import yaml

def load_config(file_path: str) -> WorkerConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkerConfig(**config_data)
