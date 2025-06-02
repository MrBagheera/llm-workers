from typing import Type, Any

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from llm_workers.api import ExtendedBaseTool


class UserInputToolSchema(BaseModel):
    prompt: str = Field(..., description="Prompt to display to the user before requesting input")


class UserInputTool(BaseTool, ExtendedBaseTool):
    name: str = "user_input"
    description: str = "Prompts the user for input and returns their response"
    args_schema: Type[UserInputToolSchema] = UserInputToolSchema
    
    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return False
    
    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return "Requesting user input"
    
    def _run(self, prompt: str) -> str:
        try:
            print(prompt)
            print("(Enter your input below, use an empty line to finish)")
            
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            return "\n".join(lines)
        except Exception as e:
            raise ToolException(f"Error reading user input: {e}")