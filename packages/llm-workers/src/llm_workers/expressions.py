import logging
import re
from typing import Any, Dict, List, Tuple, TypeVar, Generic, get_args, Literal, Optional

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from llm_workers.starlark import StarlarkEval, EvaluationContext

logger =  logging.getLogger(__name__)



class StringExpression:
    # noinspection RegExpUnnecessaryNonCapturingGroup,RegExpRedundantEscape
    _PATTERN = re.compile(r'(\\\$\{(?:.+?)\})|\$\{(.+?)\}')

    def __init__(self, value: str):
        self.raw_value = value
        self.parts: List[tuple[Literal['text'], str] | tuple[Literal['code'], StarlarkEval]] = []
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
                self.parts.append(('code', StarlarkEval(code_chunk)))
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
        script_vars: Dict[str, Any]
        if context is None:
            script_vars = {}
        elif isinstance(context, dict):
            script_vars = context
        else:
            script_vars = context.extract_all_variables()

        # --- OPTIMIZATION: Single Block Type Preservation ---
        # If the string is EXACTLY one code block with no surrounding text,
        # return the raw evaluation result (int, list, dict, etc.)
        if len(self.parts) == 1 and self.parts[0][0] == 'code':
            # noinspection PyTypeChecker
            script: StarlarkEval = self.parts[0][1]
            return script.run(script_vars, {})

        # ----------------------------------------------------

        # Standard String Interpolation
        result = []
        for kind, content in self.parts:
            if kind == 'code':
                # noinspection PyTypeChecker
                script: StarlarkEval = content
                result.append(str(script.run(script_vars, {})))
            else:
                result.append(content)

        return "".join(result)

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