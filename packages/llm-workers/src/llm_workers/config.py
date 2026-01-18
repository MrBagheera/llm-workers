from __future__ import annotations

from abc import ABC
from typing import Any, TypeAliasType, Annotated, Union, List, Optional, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, Discriminator, Tag
from pydantic import ValidationError, WrapValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import ValidatorFunctionWrapHandler, ValidationInfo

from llm_workers.expressions import StringExpression, JsonExpression


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


# noinspection PyTypeHints
Json = TypeAliasType(
    'Json',
    Annotated[
        Union[dict[str, 'Json'], list['Json'], str, int, float, bool, None],
        WrapValidator(json_custom_error_validator),
    ],
)


def create_discriminator(key_to_tag: dict[str | type, str]):
    """
    Returns a discriminator function for Pydantic.

    Args:
        key_to_tag: A dictionary mapping either:
                    1. A Type (e.g., str) -> tag
                    2. A string key (field name) -> tag
    """

    # 1. Separate keys into types (for instance checks) and strings (for dict lookups)
    type_map = {k: v for k, v in key_to_tag.items() if isinstance(k, type)}
    str_key_map = {k: v for k, v in key_to_tag.items() if isinstance(k, str)}

    valid_dict_keys = set(str_key_map.keys())

    def discriminator(v: Any) -> str | None:

        # --- Check 1: Object/Primitive Passthrough ---
        # If 'v' is a direct instance of a mapped type (e.g. str), return that tag.
        for type_cls, tag in type_map.items():
            if isinstance(v, type_cls):
                return tag

        # --- Check 2: Dictionary / JSON Parsing ---
        # If it is not a direct instance match, we check if it is a dict
        if not isinstance(v, dict):
            # If it's not a dict and didn't match our types above, return None.
            # This lets Pydantic handle the type error natively.
            return None

        # Identify which discriminator keys are present in the dict
        present_keys = [k for k in valid_dict_keys if k in v]

        # --- Strict Business Logic Errors ---
        if len(present_keys) > 1:
            raise ValueError(
                f"Ambiguous match: Found keys {present_keys}. "
                f"Only one of {list(valid_dict_keys)} allowed."
            )

        if len(present_keys) == 0:
            # Note: We raise an error here to prevent Pydantic from trying
            # other union members if the input LOOKS like a dict but is invalid.
            raise ValueError(
                f"Invalid input dictionary: Must contain one of {list(valid_dict_keys)}."
            )

        discriminator_field = present_keys[0]
        return str_key_map[discriminator_field]

    return discriminator

class RateLimiterConfig(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    requests_per_second: float
    check_every_n_seconds: float = 0.1
    max_bucket_size: float

class PricingConfig(BaseModel):
    """Pricing configuration for cost estimation based on token usage."""
    model_config = ConfigDict(extra='forbid')
    currency: str = "USD"
    input_tokens_per_million: Optional[float] = None
    output_tokens_per_million: Optional[float] = None
    cache_read_tokens_per_million: Optional[float] = None
    cache_write_tokens_per_million: Optional[float] = None

class ModelDefinition(BaseModel, ABC):
    model_config = ConfigDict(extra='forbid')
    name: str
    config: Optional[JsonExpression[dict]] = None
    rate_limiter: Optional[RateLimiterConfig] = None
    pricing: Optional[PricingConfig] = None

class StandardModelDefinition(ModelDefinition):
    provider: str
    model: str

class ImportModelDefinition(ModelDefinition):
    import_from: str


class DisplaySettings(BaseModel):
    """Display configuration settings."""
    show_token_usage: bool = True
    show_reasoning: bool = True
    auto_open_changed_files: bool = True
    markdown_output: bool = True
    file_monitor_include: list[str] = [ '*.jpg', '*.jpeg', '*.png', '*.gif', '*.tiff', '*.svg', '*.wbp' ]
    file_monitor_exclude: list[str] = ['.*', '*.log']


class EnvVarDefinition(BaseModel):
    """Definition for an environment variable."""
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    description: Optional[StringExpression] = None
    persistent: bool = False  # If false, prompted each script load
    secret: bool = False  # If true, input is hidden (requires prompt_toolkit)


class UserConfig(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    env: Optional[Dict[str, EnvVarDefinition]] = None
    models: list[StandardModelDefinition | ImportModelDefinition] = ()
    display_settings: DisplaySettings = DisplaySettings()


class MCPServerBase(BaseModel):
    """Shared attributes for all MCP servers."""
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    auto_import_scope: Literal['none', 'shared', 'chat']

class MCPServerStdio(MCPServerBase):
    transport: Literal['stdio']
    command: str
    args: JsonExpression[list] = JsonExpression([])
    env: JsonExpression[dict] = JsonExpression({})

class MCPServerHttp(MCPServerBase):
    transport: Literal['streamable_http']
    url: str
    headers: JsonExpression[dict] = JsonExpression({})

MCPServerDefinition = Annotated[
    Union[MCPServerStdio, MCPServerHttp],
    Field(discriminator='transport')
]


class EvalDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    eval: JsonExpression
    store_as: Optional[str] = None

class CallDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    call: str
    params: Optional[JsonExpression[dict]] = None
    catch: Optional[str | list[str]] = None
    store_as: Optional[str] = None
    ui_hint: Optional[StringExpression] = None

class IfDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    if_: StringExpression = Field(alias='if')
    then: 'BodyDefinition'
    else_: Optional['BodyDefinition'] = Field(default=None, alias='else')
    store_as: Optional[str] = None

class StarlarkDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    starlark: str  # The Starlark script as a string
    store_as: Optional[str] = None

class ForEachDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    for_each: JsonExpression  # Collection to iterate (list, dict, or scalar)
    do: 'BodyDefinition'      # Body to execute for each element
    store_as: Optional[str] = None


StatementDefinition = Annotated[
    Union[
        Annotated[CallDefinition, Tag('<call statement>')],
        Annotated[IfDefinition, Tag('<if statement>')],
        Annotated[EvalDefinition, Tag('<eval statement>')],
        Annotated[StarlarkDefinition, Tag('<starlark statement>')],
        Annotated[ForEachDefinition, Tag('<for_each statement>')],
    ],
    Discriminator(create_discriminator({
        'call': '<call statement>',
        CallDefinition: '<call statement>',
        'if': '<if statement>',
        IfDefinition: '<if statement>',
        'eval': '<eval statement>',
        EvalDefinition: '<eval statement>',
        'starlark': '<starlark statement>',
        StarlarkDefinition: '<starlark statement>',
        'for_each': '<for_each statement>',
        ForEachDefinition: '<for_each statement>',
    }))
]

BodyDefinition = Annotated[
    Union[
        Annotated[CallDefinition, Tag('<call statement>')],
        Annotated[IfDefinition, Tag('<if statement>')],
        Annotated[EvalDefinition, Tag('<eval statement>')],
        Annotated[StarlarkDefinition, Tag('<starlark statement>')],
        Annotated[ForEachDefinition, Tag('<for_each statement>')],
        Annotated[List[StatementDefinition], Tag('<statements>')],
    ],
    Discriminator(create_discriminator({
        'call': '<call statement>',
        CallDefinition: '<call statement>',
        'if': '<if statement>',
        IfDefinition: '<if statement>',
        'eval': '<eval statement>',
        EvalDefinition: '<eval statement>',
        'starlark': '<starlark statement>',
        StarlarkDefinition: '<starlark statement>',
        'for_each': '<for_each statement>',
        ForEachDefinition: '<for_each statement>',
        list: '<statements>'
    }))
]


class ToolDefinition(BaseModel):
    """Common fields for single tool definitions."""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Json]] = None
    return_direct: Optional[bool] = None
    confidential: Optional[bool] = None
    require_confirmation: Optional[bool] = None
    ui_hint: Optional[StringExpression | bool] = None
    ui_hint_args: List[str] = []

class ImportToolStatement(ToolDefinition):
    """Definition for an imported tool."""
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    import_tool: str

    def __str__(self):
        return f"import_tool: {self.import_tool}"


class CustomToolParamsDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    name: str
    description: str
    type: str
    default: Optional[Json] = None

class CustomToolDefinition(ToolDefinition):
    """Definition for a custom tool."""
    name: str
    input: List[CustomToolParamsDefinition] = []
    tools: list[ToolsDefinitionStatement] = []
    do: BodyDefinition

    def __str__(self):
        return f"name: {self.name}"

class ImportToolsStatement(BaseModel):
    """Statement for importing multiple tools at once."""
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    import_tools: str
    prefix: str  # mandatory prefix for tool names (can be empty "")
    filter: List[str] = ["*"]  # patterns to include/exclude tools
    force_ui_hints_for: List[str] = []  # patterns to force ui_hint=True
    force_no_ui_hints_for: List[str] = []  # patterns to force ui_hint=False
    ui_hints_args: List[str] = []  # args to include in UI hints
    force_require_confirmation_for: List[str] = []  # patterns to force require_confirmation=True
    force_no_confirmation_for: List[str] = []  # patterns to force require_confirmation=False

    @property
    def import_tools_split(self) -> (str, str):
        if ':' in self.import_tools:
            i = self.import_tools.index(':')
            return self.import_tools[:i], self.import_tools[i + 1:]
        return '', self.import_tools

# For defining tools - single or multiple.
# This incantation allows us to pick concrete Type based on field presence,
# and get nicer error messages if validation fails (compared to using validators).
ToolsDefinitionStatement = Annotated[
    Union[
        Annotated[ImportToolStatement, Tag('<import_tool statement>')],
        Annotated[ImportToolsStatement, Tag('<import_tools statement>')],
        Annotated[CustomToolDefinition, Tag('<custom_tool definition>')],
    ],
    Discriminator(create_discriminator({
        'import_tool': '<import_tool statement>',
        'import_tools': '<import_tools statement>',
        'do': '<custom_tool definition>',
    }))
]


class ToolsReference(BaseModel):
    match: list[str]

# For referencing multiple tools.
# This incantation allows us to pick concrete Type based on field presence,
# and get nicer error messages if validation fails (compared to using validators).
ToolsDefinitionOrReference = Annotated[
    Union[
        Annotated[str, Tag('<tool reference>')],
        Annotated[ToolsReference, Tag('<tools references>')],
        Annotated[ImportToolStatement, Tag('<import_tool statement>')],
        Annotated[ImportToolsStatement, Tag('<import_tools statement>')],
        Annotated[CustomToolDefinition, Tag('<custom_tool definition>')],
    ],
    Discriminator(create_discriminator({
        str: '<tool reference>',
        'match': '<tools references>',
        'import_tool': '<import_tool statement>',
        'import_tools': '<import_tools statement>',
        'do': '<custom_tool definition>',
    }))
]


class BaseLLMConfig(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    model_ref: str = "default"
    system_message: Optional[StringExpression] = None
    tools: List[ToolsDefinitionOrReference] = []


class ToolLLMConfig(BaseLLMConfig):
    extract_json: Optional[Union[bool, str]] = None


class SharedSectionConfig(BaseModel):
    """Configuration for shared data and tools."""
    model_config = ConfigDict(extra='forbid')
    data: dict[str, JsonExpression] = {}
    tools: list[ToolsDefinitionStatement] = []


class ChatConfig(BaseLLMConfig):
    model_config = ConfigDict(extra='forbid')
    default_prompt: Optional[str] = None
    user_banner: Optional[str] = None


class CliConfig(BaseModel):
    """Configuration for CLI workers."""
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    process_input: Literal['one_by_one'] | Literal['all_as_list']
    tools: list[ToolsDefinitionOrReference] = []
    json_output: bool | Literal['pretty'] = False
    do: BodyDefinition


class WorkersConfig(BaseModel):
    model_config = ConfigDict(extra='forbid') # Forbid extra fields to ensure strictness
    env: Optional[Dict[str, EnvVarDefinition]] = None
    mcp: Dict[str, MCPServerDefinition] = {}
    shared: SharedSectionConfig = SharedSectionConfig()
    chat: Optional[ChatConfig] = None
    cli: Optional[CliConfig] = None
