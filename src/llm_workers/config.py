from __future__ import annotations

import importlib.resources
from abc import ABC
from typing import Any, TypeAliasType, Annotated, Union, List, Optional, Dict

import yaml
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, model_validator, PrivateAttr, ConfigDict
from pydantic import ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo
from typing_extensions import Self


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


class RateLimiterConfig(BaseModel):
    requests_per_second: float
    check_every_n_seconds: float = 0.1
    max_bucket_size: float

class ModelDefinition(BaseModel, ABC):
    model_config = ConfigDict(extra="allow")
    name: str
    config: Optional[Dict[str, Json]] = None
    rate_limiter: Optional[RateLimiterConfig] = None

class StandardModelDefinition(ModelDefinition):
    provider: str
    model: str

class ImportModelDefinition(ModelDefinition):
    import_from: str


class DisplaySettings(BaseModel):
    """Display configuration settings."""
    show_token_usage: bool = True
    show_reasoning: bool = False
    auto_open_changed_files: bool = False
    markdown_output: bool = True
    file_monitor_include: list[str] = [ '*.jpg', '*.jpeg', '*.png', '*.gif', '*.tiff', '*.svg', '*.wbp' ]
    file_monitor_exclude: list[str] = ['.*', '*.log']


class UserConfig(BaseModel):
    models: list[StandardModelDefinition | ImportModelDefinition] = ()
    display_settings: DisplaySettings = DisplaySettings()


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
    key: Optional[Json] = None
    default: Optional[Json] = None

class CallDefinition(BaseModel):
    call: ToolReference
    params: Optional[Dict[str, Json]] = None
    catch: Optional[str | list[str]] = None

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


def _ensure_only_one_of(values: dict[str, Any], keys: set[str], context: str):
    """Ensure that only one of the specified parameters is present in the values."""
    if sum(1 for key in keys if key in values) > 1:
        raise ValueError(f"Only one of {keys} should be specified in {context}.")

def _ensure_set(model: Any, keys: list[str], context: str):
    """Ensure that the specified parameters are set in the model."""
    violations = [param for param in keys if getattr(model, param) is None]
    if len(violations) > 0:
        raise ValueError(f"Required fields {violations} are missing in {context}.")

def _ensure_not_set(model: Any, keys: list[str], context: str):
    """Ensure that the specified parameters are set in the model."""
    violations = [param for param in keys if getattr(model, param) is not None]
    if len(violations) > 0:
        raise ValueError(f"Fields {violations} are not supported in {context}.")

class CustomToolDefinition(BaseModel):
    input: List[CustomToolParamsDefinition] = []
    body: BodyDefinition

class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: Optional[str] = None
    config: Optional[Dict[str, Json]] = None
    return_direct: Optional[bool] = None
    confidential: Optional[bool] = None
    require_confirmation: Optional[bool] = None
    ui_hint: Optional[str] = None
    _ui_hint_template: Optional[PromptTemplate] = PrivateAttr(default=None)  # private field
    import_from: Optional[str] = None  # for imported tools, the symbol to import from

    def __init__(self, **data):
        super().__init__(**data)
        if self.ui_hint is not None:
            self._ui_hint_template = PromptTemplate.from_template(self.ui_hint)

    @property
    def ui_hint_template(self) -> Optional[PromptTemplate]:
        return self._ui_hint_template


ToolReference = TypeAliasType(
    'ToolReference',
    Union[str, ToolDefinition],
)


class BaseLLMConfig(BaseModel):
    model_ref: str = "default"
    system_message: str = None
    tools: Optional[List[ToolReference]] = None


class ToolLLMConfig(BaseLLMConfig):
    extract_json: Optional[Union[bool, str]] = None


class ChatConfig(BaseLLMConfig):
    default_prompt: Optional[str] = None
    user_banner: Optional[str] = None


class WorkersConfig(BaseModel):
    tools: list[ToolDefinition] = ()
    shared: Dict[str, Json] = {}
    chat: Optional[ChatConfig] = None
    cli: Optional[BodyDefinition] = None


def load_config(name: str) -> WorkersConfig:
    # if name has module:resource format, load it as a module
    if ':' in name:
        module, resource = name.split(':', 1)
        if len(module) > 1: # ignore volume names on windows
            with importlib.resources.files(module).joinpath(resource).open("r") as file:
                config_data = yaml.safe_load(file)
            return WorkersConfig(**config_data)
    # try loading as file
    with open(name, 'r') as file:
        config_data = yaml.safe_load(file)
    return WorkersConfig(**config_data)
