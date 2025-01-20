from typing import List, Dict

from pydantic import BaseModel


class ToolParams(BaseModel):
    description: str
    type: str
    optional: bool

class Tool(BaseModel):
    type: str
    description: str
    params: Dict[str, ToolParams]
    instructions: str

class CustomTool(BaseModel):
    type: str
    description: str
    params: Dict[str, ToolParams]
    tool_refs: List[str]
    instructions: str

class Main(BaseModel):
    instructions: str
    interactive: bool
    tool_refs: List[str]

class MetacriticMonkeyConfig(BaseModel):
    custom_tools: List[CustomTool]
    main: Main


import yaml

def load_config(file_path: str) -> MetacriticMonkeyConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return MetacriticMonkeyConfig(**config_data)
