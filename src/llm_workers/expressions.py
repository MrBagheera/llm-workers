import getpass
import json
import logging
import os
import platform
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypeVar, Generic, get_args, Literal, Optional

import simpleeval
from langchain_core.tools import ToolException
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from simpleeval import SimpleEval

from llm_workers.utils import LazyFormatter

logger =  logging.getLogger(__name__)


def get_with_default(container, key, default):
    if isinstance(container, list):
        return container[key] if key <=0 and key < len(container) else default
    if isinstance(container, dict):
        return container.get(key, default)
    raise ValueError(f"{type(container)} is not allowed container")

def merge(arg1, arg2):
    """Merge two arguments into one."""
    if isinstance(arg1, list) and isinstance(arg2, list):
        return arg1 + arg2
    if isinstance(arg1, dict) and isinstance(arg2, dict):
        result = arg1.copy()
        result.update(arg2)
        return result
    return str(arg1) + str(arg2)

def split(arg: str, delimiter: str) -> List[str]:
    """Split a string into a list based on the given delimiter."""
    return arg.split(delimiter)

def join(arg: List[str], delimiter: str) -> str:
    """Join a list of strings into a single string with the given delimiter."""
    return delimiter.join(arg)

def strip(arg: str) -> str:
    """Strip whitespace from both ends of the string."""
    return arg.strip()

def flatten(arg: List[Any]) -> List[Any]:
    """Flatten a nested list into a single list."""
    result = []
    for item in arg:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def parse_json(arg: str, ignore_error: bool = False) -> Any:
    """Parse a JSON string into a Python object."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        if ignore_error:
            return arg
        raise simpleeval.InvalidExpression(f'Failed to parse JSON from: {LazyFormatter(arg)}')

def print_json(arg: Any) -> str:
    """Convert a Python object into a JSON string."""
    return json.dumps(arg)

def is_string(value: Any) -> bool:
    return isinstance(value, str)

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))

def is_list(value: Any) -> bool:
    return isinstance(value, list)

def is_dict(value: Any) -> bool:
    return isinstance(value, dict)

def is_bool(value: Any) -> bool:
    return isinstance(value, bool)

DEFAULT_FUNCTIONS = simpleeval.DEFAULT_FUNCTIONS.copy()
DEFAULT_FUNCTIONS['len'] = len  # Ensure len() is available
DEFAULT_FUNCTIONS['get'] = get_with_default
DEFAULT_FUNCTIONS['merge'] = merge
DEFAULT_FUNCTIONS['split'] = split
DEFAULT_FUNCTIONS['join'] = join
DEFAULT_FUNCTIONS['strip'] = strip
DEFAULT_FUNCTIONS['flatten'] = flatten
DEFAULT_FUNCTIONS['parse_json'] = parse_json
DEFAULT_FUNCTIONS['print_json'] = print_json
DEFAULT_FUNCTIONS['is_string'] = is_string
DEFAULT_FUNCTIONS['is_number'] = is_number
DEFAULT_FUNCTIONS['is_list'] = is_list
DEFAULT_FUNCTIONS['is_dict'] = is_dict
DEFAULT_FUNCTIONS['is_bool'] = is_bool

FIXED_NAMES = {
    "True": True,
    "true": True,
    "False": False,
    "false": False,
    "None": None,
}

class EvaluationContext:
    """
    Context for evaluating expressions.
    Holds variable bindings.
    """
    def __init__(self, variables: Dict[str, Any] = None, parent: 'EvaluationContext' = None, mutable: bool = True):
        self.parent = parent
        self.variables = variables or {}
        self.mutable = mutable

    def get(self, name: str) -> Any:
        if name in FIXED_NAMES:
            return FIXED_NAMES[name]
        if name in self.variables:
            return self.variables[name]
        else:
            p = self.parent
            while p:
                if name in p.variables:
                    return p.variables[name]
                p = p.parent
            return None

    @property
    def known_names(self) -> List[str]:
        result = list(self.variables.keys())
        if self.parent:
            result += list(self.parent.variables.keys())
        return result

    def resolve(self, node: Any) -> Any:
        name = node.id
        value = self.get(name)
        if value is not None:
            return value
        raise simpleeval.InvalidExpression(f"'{name}' is not defined, available names: {self.known_names}")

    def add(self, name: str, value: Any):
        if not self.mutable:
            raise RuntimeError(f"Cannot add variable {name} to immutable EvaluationContext")
        self.variables[name] = value

    @staticmethod
    def default_environment() -> Dict[str, Any]:
        os_name = platform.system()
        return {
            "UserName": getpass.getuser(),
            "OS": os_name,
            "CurrentDate": datetime.now().strftime('%Y-%m-%d'),
            "WorkDir": os.getcwd(),
        }

class StringExpression:
    # noinspection RegExpUnnecessaryNonCapturingGroup,RegExpRedundantEscape
    _PATTERN = re.compile(r'(\\\$\{(?:.+?)\})|\$\{(.+?)\}')

    def __init__(self, value: str):
        self.raw_value = value
        self.parts: List[tuple[Literal['text'], str] | tuple[Literal['code'], tuple[str, any]]] = []
        self.is_dynamic = False
        self._static_value: Optional[str] = None    # may differ from raw_value due to un-escaping
        self._parse_value()

    def _parse_value(self):
        """
        Parses the string into text and code parts.
        """
        tokens = self._PATTERN.split(self.raw_value)
        current_text = []

        # Iterate through regex split results
        for i in range(0, len(tokens), 3):
            text_chunk = tokens[i]
            escaped_chunk = tokens[i+1] if i+1 < len(tokens) else None
            code_chunk = tokens[i+2] if i+2 < len(tokens) else None

            if text_chunk:
                current_text.append(text_chunk)

            if escaped_chunk:
                # Strip backslash from escaped blocks (e.g. "\${val}" -> "${val}")
                current_text.append(escaped_chunk[1:])

            if code_chunk:
                # Flush existing text if any
                if current_text:
                    self.parts.append(('text', "".join(current_text)))
                    current_text = []
                self.parts.append(('code', (code_chunk, SimpleEval.parse(code_chunk))))
                self.is_dynamic = True

        # Flush trailing text
        if current_text:
            self.parts.append(('text', "".join(current_text)))

        # Pre-calculate static value if no code blocks exist
        if not self.is_dynamic:
            self._static_value = "".join(part[1] for part in self.parts)

    def evaluate(self, context: EvaluationContext | Dict[str, Any] = None) -> Any:
        """
        Evaluates the expression.
        - If static: returns string.
        - If single code block (e.g. "${val}"): returns raw type (int, dict, etc).
        - If mixed (e.g. "Val: ${val}"): returns string.
        """
        if not self.is_dynamic:
            return self._static_value

        # Accept both dict and EvaluationContext for convenience
        if context is None:
            context = EvaluationContext()
        elif isinstance(context, dict):
            context = EvaluationContext(context)

        # --- OPTIMIZATION: Single Block Type Preservation ---
        # If the string is EXACTLY one code block with no surrounding text,
        # return the raw evaluation result (int, list, dict, etc.)
        if len(self.parts) == 1 and self.parts[0][0] == 'code':
            return self._eval(self.parts[0][1], context)

        # ----------------------------------------------------

        # Standard String Interpolation
        result = []
        for kind, content in self.parts:
            if kind == 'code':
                try:
                    result.append(str(self._eval(content, context)))
                except Exception as e:
                    # Most likely calling LLM cannot do anything with this error,
                    # so we log it as warning to simplify debugging
                    message = f"Failed to evaluate ${{{content}}}: {e}"
                    logger.warning(message)
                    raise ToolException(message)
            else:
                result.append(content)

        return "".join(result)

    @staticmethod
    def _eval(code: tuple[str, any], context: EvaluationContext) -> Any:
        expr, parsed = code
        try:
            s = SimpleEval(functions=DEFAULT_FUNCTIONS, names=context.resolve)
            return s.eval(expr, parsed)
        except Exception as e:
            # Most likely calling LLM cannot do anything with this error,
            # so we log it as warning to simplify debugging
            message = f"Failed to evaluate ${{{expr}}}: {e}"
            logger.warning(message)
            raise ToolException(message)

    def __str__(self):
        return self.raw_value

    def __repr__(self):
        return f"StringExpression('{self.raw_value}')"

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema([
            # 1. Accept an existing instance (e.g., from Python code)
            core_schema.is_instance_schema(cls),

            # 2. Accept a raw string (e.g., from JSON) -> validate -> convert
            core_schema.no_info_after_validator_function(
                cls,
                core_schema.str_schema()
            )
        ])


T = TypeVar("T")

class JsonExpression(Generic[T]):
    """
    A generic class that parses JSON with support for ${...} expressions.
    Usage:
      field: JsonExpression[dict]  # Enforces JSON object
      field: JsonExpression[list]  # Enforces JSON array
      field: JsonExpression        # Accepts Any JSON
    """
    def __init__(self, value: T):
        self._raw_value = value
        self._structure: Any = None
        self._is_dynamic = False

        # Parse the structure immediately
        self._structure, self._is_dynamic = self._parse_structure(value)

    def evaluate(self, context: EvaluationContext | Dict[str, Any] = None) -> T:
        """
        Returns the evaluated structure with correct type hint T.
        """
        if not self._is_dynamic:
            return self._raw_value

        # Accept both dict and EvaluationContext for convenience
        if context is None:
            context = EvaluationContext()
        elif isinstance(context, dict):
            context = EvaluationContext(context)
        return self._eval_node(self._structure, context)

    def _parse_structure(self, node: Any) -> Tuple[Any, bool]:
        """
        Recursively identifies dynamic parts of the JSON graph.
        Returns: (processed_node, is_dynamic)
        """
        if isinstance(node, str):
            expr = StringExpression(node)
            # It's dynamic if it has code blocks OR if unescaping changed the string
            if expr.is_dynamic or expr.evaluate({}) != node:
                return expr, True
            return node, False

        elif isinstance(node, dict):
            new_dict = {}
            any_dynamic = False
            for k, v in node.items():
                processed_v, v_dynamic = self._parse_structure(v)
                new_dict[k] = processed_v
                if v_dynamic:
                    any_dynamic = True

            # Optimization: preserve original ref if fully static
            if not any_dynamic:
                return node, False
            return new_dict, True

        elif isinstance(node, list):
            new_list = []
            any_dynamic = False
            for item in node:
                processed_item, item_dynamic = self._parse_structure(item)
                new_list.append(processed_item)
                if item_dynamic:
                    any_dynamic = True

            if not any_dynamic:
                return node, False
            return new_list, True

        else:
            # Primitives (int, float, bool, None)
            return node, False

    def _eval_node(self, node: Any, context: EvaluationContext) -> Any:
        """
        Recursively evaluates the parsed structure.
        """
        if isinstance(node, StringExpression):
            return node.evaluate(context)
        elif isinstance(node, dict):
            return {k: self._eval_node(v, context) for k, v in node.items()}
        elif isinstance(node, list):
            return [self._eval_node(i, context) for i in node]
        return node

    def __repr__(self):
        return f"JsonExpression({self._raw_value})"

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Determine the inner schema (e.g. dict, list, or Any)
        inner_schema = core_schema.any_schema()
        args = get_args(source_type)
        if args:
            inner_schema = handler.generate_schema(args[0])

        return core_schema.union_schema([
            # 1. Accept an existing instance
            core_schema.is_instance_schema(cls),

            # 2. Accept raw data -> validate structure -> convert
            core_schema.no_info_after_validator_function(
                cls,
                inner_schema
            )
        ])