import hashlib
import json
import time
from random import random
from typing import Type, Any, Dict

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from llm_workers.api import ConfirmationRequest, ExtendedBaseTool, ConfirmationRequestParam
from llm_workers.config import Json

# Module-local dictionary to store approval tokens
_approval_tokens: Dict[str, Dict[str, Json]] = {}

def store_approval_token(token: str, data: Json) -> None:
    """Store approval token in module-local dictionary."""
    _approval_tokens[token] = {
        "token": token,
        "created_at": time.time(),
        "data": data
    }

def generate_and_store_approval_token(data: Json) -> str:
    """Generate a new random approval token, store it with the data, and return the token."""
    rnd: float = random()
    token = hashlib.sha256(str(rnd).encode()).hexdigest()
    store_approval_token(token, data)
    return json.dumps({"approval_token": token})

def validate_approval_token(token: str) -> Json:
    """Validate that approval token exists."""
    token_data = _approval_tokens.get(token)
    if token_data is None:
        raise ToolException(f"Invalid or already consumed approval token: {token}")
    return token_data["data"]

def consume_approval_token(token: str) -> bool:
    """Consume approval token. Returns True if token was valid and consumed."""
    if token not in _approval_tokens:
        return False
    _approval_tokens.pop(token)
    return True


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


class RequestApprovalToolSchema(BaseModel):
    prompt: str = Field(..., description="Prompt to show to user for approval")


class RequestApprovalTool(BaseTool, ExtendedBaseTool):
    """Tool that requests user approval and returns an approval token."""
    
    name: str = "request_approval"
    description: str = "Request approval from user and return approval token"
    args_schema: Type[RequestApprovalToolSchema] = RequestApprovalToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return True

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        prompt = input["prompt"]
        return ConfirmationRequest(
            action='get approval for following actions',
            params=[ConfirmationRequestParam(name="action", value=prompt, format="markdown")],
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return ""

    def _run(self, prompt: str) -> str:
        return generate_and_store_approval_token(prompt)


class ValidateApprovalToolSchema(BaseModel):
    approval_token: str = Field(..., description="Approval token to validate")


class ValidateApprovalTool(BaseTool, ExtendedBaseTool):
    """Tool that validates an approval token exists and is not consumed."""
    
    name: str = "validate_approval"
    description: str = "Validate approval token exists and is not consumed"
    args_schema: Type[ValidateApprovalToolSchema] = ValidateApprovalToolSchema

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return ""

    def _run(self, approval_token: str) -> str:
        data = validate_approval_token(approval_token)
        return data if isinstance(data, str) else json.dumps(data)


class ConsumeApprovalToolSchema(BaseModel):
    approval_token: str = Field(..., description="Approval token to consume")


class ConsumeApprovalTool(BaseTool, ExtendedBaseTool):
    """Tool that validates and consumes an approval token, making it unusable."""
    
    name: str = "consume_approval"
    description: str = "Validate and consume approval token, making it unusable"
    args_schema: Type[ConsumeApprovalToolSchema] = ConsumeApprovalToolSchema

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return ""

    def _run(self, approval_token: str) -> str:
        was_consumed = consume_approval_token(approval_token)
        if not was_consumed:
            raise ToolException(f"Invalid or already consumed approval token: {approval_token}")
        return "Approval token consumed"
