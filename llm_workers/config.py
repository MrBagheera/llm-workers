from typing import Any, TypeAliasType, Annotated, Union, List, Optional, Dict

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo
from typing_extensions import Self

from llm_workers.llm import BaseLLMConfig


def json_custom_error_validator(
        value: Any,
        handler: ValidatorFunctionWrapHandler,
        _info: ValidationInfo
) -> Any:
    """Simplify the error message to avoid a gross error stemming
    from exhaustive checking of all union options.
    """
    try:
        return handler(value)
    except ValidationError:
        raise PydanticCustomError(
            'invalid_json',
            'Input is not valid json',
        )

Json = TypeAliasType(
    'Json',
    Annotated[
        Union[dict[str, 'Json'], list['Json'], str, int, float, bool, None],
        WrapValidator(json_custom_error_validator),
    ],
)


class ModelConfig(BaseModel):
    name: str
    provider: str
    model: str
    model_params: Json = None


StatementDefinition = TypeAliasType(
    'StatementDefinition',
    Union['CallDefinition', 'MatchDefinition', 'ResultDefinition'],
)

BodyDefinition = TypeAliasType(
    'BodyDefinition',
    Union[StatementDefinition, List[StatementDefinition]],
)

class ResultDefinition(BaseModel):
    result: Json

class CallDefinition(BaseModel):
    call: str
    params: Optional[Dict[str, Json]] = None
    model_config = ConfigDict(extra='allow')

class MatchClauseDefinition(BaseModel):
    case: Optional[str] = None
    pattern: Optional[str] = None
    then: BodyDefinition

    @classmethod
    @model_validator(mode='after')
    def validate(cls, value: Any) -> Self:
        if value.case is None and value.pattern is None:
            raise ValueError("Either 'case' or 'pattern' must be provided")
        if value.case is not None and value.pattern is not None:
            raise ValueError("Only one of 'case' or 'pattern' can be provided")
        return value

class MatchDefinition(BaseModel):
    match: str
    trim: bool = False
    matchers: List[MatchClauseDefinition]
    default: BodyDefinition

class CustomToolParamsDefinition(BaseModel):
    name: str
    description: str
    type: str
    default: Optional[Json] = None

class CustomToolDefinition(BaseModel):
    name: str
    description: str
    input: List[CustomToolParamsDefinition]
    body: BodyDefinition
    return_direct: bool = False


class ChatConfig(BaseLLMConfig):
    default_prompt: Optional[str] = None

class WorkerConfig(BaseModel):
    models: list[ModelConfig]
    custom_tools: list[CustomToolDefinition] = ()
    chat: Optional[ChatConfig] = None
    cli: Optional[BodyDefinition] = None


def load_config(file_path: str) -> WorkerConfig:
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkerConfig(**config_data)
