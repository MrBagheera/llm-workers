from typing import Literal, Annotated

import requests
from langchain_core.tools import InjectedToolArg, ToolException, BaseTool
from pydantic import BaseModel
from requests import RequestException

from llm_workers.tools.custom_tools_base import CustomToolBaseDefinition, Json, TemplateHelper, build_dynamic_tool
from llm_workers.utils import ensure_environment_variable


class T2AiWrapperToolDefinition(CustomToolBaseDefinition):
    type: Literal['t2-ai-wrapper']
    endpoint: Annotated[str, InjectedToolArg]
    payload: Annotated[Json, InjectedToolArg] = None


# Auth token to include in AI Relay API calls
_auth_token = None


def _call_t2_ai_wrapper(endpoint, payload) -> Json:
    """
    Calls T2AI wrapper with given endpoint and payload.
    This tool is not supposed to be called by LLM directly,
    only via tool binding.

    See https://wiki.corp.zynga.com/pages/viewpage.action?pageId=261429979

    Args:
        endpoint: text endpoint URL
        payload: JSON payload for POST request
    """
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + _auth_token
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    # Check for a valid response (HTTP Status Code 200)
    if response.status_code != 200:
            raise RequestException(f"HTTP status code {response.status_code}")
    result = response.json()

    # if endpoint ends with '/chat'
    if endpoint.endswith('/chat/prompt'):
        output = result.get('data', {}).get('output')
        if output:
            return output
        else:
            raise ToolException(f"Unexpected response from T2 AI wrapper: {result}")
    else:
        outputs = result.get('data', {}).get('outputs')
        if isinstance(outputs, list):
            return outputs[0]
        else:
            raise ToolException(f"Unexpected response from T2 AI wrapper: {result}")

def build_t2_ai_wrapper(definition: T2AiWrapperToolDefinition) -> BaseTool:
    global _auth_token
    _auth_token = ensure_environment_variable("T2_AI_TOKEN")
    template_helper = TemplateHelper(definition, definition.payload)

    def tool_logic(validated_input: BaseModel):
        target_payload = template_helper.render(validated_input.model_dump())
        return _call_t2_ai_wrapper(definition.endpoint, target_payload)

    return build_dynamic_tool(definition, tool_logic, async_tool_logic = None)
