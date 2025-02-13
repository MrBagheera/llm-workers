from typing import Annotated

import requests
from langchain_core.tools import InjectedToolArg, ToolException, StructuredTool
from requests import RequestException

from llm_workers.config import Json
from llm_workers.utils import ensure_environment_variable

# Auth token to include in AI Relay API calls
_auth_token = ensure_environment_variable("T2_AI_TOKEN")


def _call_t2_ai_wrapper(
    endpoint: Annotated[str, InjectedToolArg],
    payload: Annotated[Json, InjectedToolArg]
) -> Json:
    """
    Calls T2 AI Relay with given endpoint and payload.
    This tool is not supposed to be called by LLM directly,
    only via custom tool.

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

t2_ai_wrapper = StructuredTool.from_function(
    _call_t2_ai_wrapper,
    name="t2_ai_wrapper",
    parse_docstring=True,
    error_on_invalid_docstring=True
)
