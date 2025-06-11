import hashlib
import json
import time
from typing import Type, Any, Dict

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from llm_workers.api import ConfirmationRequest, ExtendedBaseTool, ConfirmationRequestParam

# Module-local dictionary to store approval tokens
_approval_tokens: Dict[str, Dict[str, Any]] = {}


def _generate_approval_token(prompt: str) -> str:
    """Generate a unique approval token based on prompt and timestamp."""
    timestamp = str(time.time())
    content = f"{prompt}:{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()


def _store_approval_token(token: str) -> None:
    """Store approval token in module-local dictionary."""
    _approval_tokens[token] = {
        "token": token,
        "created_at": time.time(),
        "used": False
    }


def _validate_approval_token(token: str) -> bool:
    """Validate that approval token exists and is not consumed."""
    token_data = _approval_tokens.get(token)
    if token_data is None:
        return False
    return token_data.get("used", True) is False


def _consume_approval_token(token: str) -> bool:
    """Mark approval token as consumed. Returns True if token was valid and consumed."""
    token_data = _approval_tokens.get(token)
    if token_data is None:
        return False
    
    if token_data.get("used", True) is True:
        return False
    
    token_data["used"] = True
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
            action='approve following actions',
            params=[ConfirmationRequestParam(name="action", value=prompt, format="markdown")],
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return ""

    def _run(self, prompt: str) -> str:
        try:
            token = _generate_approval_token(prompt)
            _store_approval_token(token)
            return json.dumps({"approval_token": token})
        except Exception as e:
            raise ToolException(f"Error generating approval token: {e}")


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
        try:
            is_valid = _validate_approval_token(approval_token)
            if not is_valid:
                raise ToolException(f"Invalid or already consumed approval token: {approval_token}")
            return "Approval token is valid"
        except Exception as e:
            raise ToolException(f"Error validating approval token: {e}")


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
        try:
            was_consumed = _consume_approval_token(approval_token)
            if not was_consumed:
                raise ToolException(f"Invalid or already consumed approval token: {approval_token}")
            return "Approval token consumed successfully"
        except Exception as e:
            raise ToolException(f"Error consuming approval token: {e}")