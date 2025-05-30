import importlib.resources
from abc import ABC
from typing import Any, TypeAliasType, Annotated, Union, List, Optional, Dict

import yaml
from pydantic import BaseModel, model_validator, Field, PrivateAttr
from pydantic import ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo
from typing_extensions import Self

from langchain_core.prompts import PromptTemplate

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

class ModelConfig(BaseModel, ABC):
    name: str
    model_params: Json = None
    rate_limiter: Optional[RateLimiterConfig] = None

class StandardModelConfig(ModelConfig):
    provider: str
    model: str

class ImportModelConfig(ModelConfig):
    import_from: str


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


def _ensure_only_one_of(values: dict[str, any], keys: set[str], context: str):
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

class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input: Optional[List[CustomToolParamsDefinition]] = None # only for custom tools
    tool_config: Optional[dict[str, Json]] = None # only for imported tools
    return_direct: Optional[bool] = None
    confidential: Optional[bool] = None
    require_confirmation: Optional[bool] = None
    ui_hint: Optional[str] = None
    _ui_hint_template: Optional[PromptTemplate] = PrivateAttr(default=None)  # private field
    # actual implementation definition (only one of these)
    clazz: Optional[str] = Field(alias='class', default=None)
    factory: Optional[str] = None
    body: Optional[BodyDefinition] = None

    @classmethod
    @model_validator(mode="wrap")
    def validate_wrapper(cls, values, handler):
        _ensure_only_one_of(values, {'clazz', 'factory', 'body'}, 'tool definition')
        model = handler(values)  # Calls the standard validation process
        if model.body is not None: # custom tool
            _ensure_set(model, ['description', 'input'], 'custom tool definition')
            _ensure_not_set(model, ['tool_config'], 'custom tool definition')
        else: # imported tool
            _ensure_not_set(model, ['input'], 'imported tool definition')
        return model

    def __init__(self, **data):
        super().__init__(**data)
        if self.ui_hint is not None:
            self._ui_hint_template = PromptTemplate.from_template(self.ui_hint)

    @property
    def ui_hint_template(self) -> Optional[PromptTemplate]:
        return self._ui_hint_template

class BaseLLMConfig(BaseModel):
    model_ref: str = "default"
    system_message: str = None
    tool_refs: Optional[List[str]] = None
    remove_past_reasoning: bool = False # experimental - to test on bigger chats.
                                        # If it proves to bring no benefits, remove it


class ChatConfig(BaseLLMConfig):
    default_prompt: Optional[str] = None
    user_banner: Optional[str] = None
    show_reasoning: bool = False
    auto_open_changed_files: bool = False
    file_monitor_include: list[str] = ['*']
    file_monitor_exclude: list[str] = ['.*', '*.log']


class WorkersConfig(BaseModel):
    models: list[StandardModelConfig | ImportModelConfig] = ()
    tools: list[ToolDefinition] = ()
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
