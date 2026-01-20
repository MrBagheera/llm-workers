import ast
import getpass
import json
import os
import platform
from datetime import datetime
from types import CodeType
from typing import Dict, Any, Optional, Callable, Literal, List

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence, guarded_unpack_sequence
from langchain_core.tools import ToolException

from llm_workers.utils import LazyFormatter


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
            result.extend(self.parent.known_names)
        return result

    def add(self, name: str, value: Any):
        if not self.mutable:
            if name in self.variables:
                raise RuntimeError(f"Cannot modify existing variable {name} in immutable EvaluationContext")
            if self.parent is not None:
                self.parent.add(name, value)
            else:
                raise RuntimeError(f"Cannot add variable {name} to immutable EvaluationContext")
        self.variables[name] = value

    def extract_all_variables(self) -> Dict[str, Any]:
        """Recursively extract all variables from context including parents."""
        result = {}
        # Start with parent variables (so child can override)
        if self.parent:
            result.update(self.parent.extract_all_variables())
        # Add local variables (override parent values)
        result.update(self.variables)
        return result

    @staticmethod
    def default_environment() -> Dict[str, Any]:
        os_name = platform.system()
        return {
            "UserName": getpass.getuser(),
            "OS": os_name,
            "CurrentDate": datetime.now().strftime('%Y-%m-%d'),
            "WorkDir": os.getcwd(),
        }

# --- Custom AST Validators ---

class StarlarkSyntaxValidator(ast.NodeVisitor):
    def visit_While(self, node):
        raise SyntaxError("Starlark does not support 'while' loops.")
    def visit_ClassDef(self, node):
        raise SyntaxError("Starlark does not support class definitions (use structs).")
    def visit_Import(self, node):
        raise SyntaxError("Starlark does not support 'import' (use load()).")
    def visit_ImportFrom(self, node):
        raise SyntaxError("Starlark does not support 'import' (use load()).")
    def visit_Try(self, node):
        raise SyntaxError("Starlark does not support try/except blocks.")
    def visit_Yield(self, node):
        raise SyntaxError("Starlark does not support generators/yield.")

    # Eval-specific blocks (handled conditionally later if needed, but good defaults)
    def visit_NamedExpr(self, node):
        raise SyntaxError("Starlark does not support assignment expressions (':=').")

class RecursionValidator(ast.NodeVisitor):
    def __init__(self):
        self.current_function = None

    def visit_FunctionDef(self, node):
        if self.current_function is not None:
            raise SyntaxError(f"Starlark does not support nested functions ('{node.name}').")

        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and self.current_function:
            if node.func.id == self.current_function:
                raise SyntaxError(f"Recursion detected: Function '{self.current_function}' calls itself.")
        self.generic_visit(node)

# --- Runtime Helpers ---

class StarlarkStruct:
    """A simple data container behaving like a Starlark struct."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        items = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        return f"struct({items})"
    def __eq__(self, other):
        return isinstance(other, StarlarkStruct) and self.__dict__ == other.__dict__

def _starlark_getattr(obj, name):
    if name.startswith('_'):
        raise AttributeError(f"Access to private attribute '{name}' is forbidden.")
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
    return getattr(obj, name)

def _sanitize_data(obj):
    """Recursively converts Python objects into Starlark-safe primitives."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_data(item) for item in obj]
    if hasattr(obj, '__dict__'):
        safe_data = {k: _sanitize_data(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        return StarlarkStruct(**safe_data)
    return str(obj)

def _parse_json(arg: str, ignore_error: bool = False) -> Any:
    """Parse a JSON string into a Python object."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        if ignore_error:
            return arg
        raise ValueError(f'Failed to parse JSON from: {LazyFormatter(arg)}')

def _print_json(arg: Any) -> str:
    """Convert a Python object into a JSON string."""
    return json.dumps(arg, ensure_ascii=False)

# --- Base Class ---

class StarlarkBase:
    def __init__(self, code_string: str, mode: Literal['eval', 'exec']):
        self.code_string = code_string
        self.mode = mode
        self.bytecode: Optional[CodeType] = None

        # 1. Prepare Builtins
        self.builtins = safe_builtins.copy()
        # Remove unsafe Python builtins
        blocked = ['__import__', 'compile', 'delattr', 'eval', 'exec', 'globals',
                   'help', 'input', 'locals', 'memoryview', 'open', 'super', 'vars', 'print']
        for name in blocked:
            self.builtins.pop(name, None)
        # Add Starlark basics
        self.builtins['struct'] = StarlarkStruct
        # Add own build-ins
        self.builtins['parse_json'] = _parse_json
        self.builtins['print_json'] = _print_json

        # 2. Compile immediately (Fail fast)
        self._compile()

    def _compile(self):
        # AST Validation
        try:
            tree = ast.parse(self.code_string, mode=self.mode)
        except SyntaxError as e:
            raise SyntaxError(f"Starlark Syntax Error: {e.text}")

        # Check syntax restrictions
        StarlarkSyntaxValidator().visit(tree)

        # Check recursion (only relevant if functions are defined, mostly for 'exec')
        if self.mode == 'exec':
            RecursionValidator().visit(tree)
        elif self.mode == 'eval':
            # Additional eval specific checks
            for node in ast.walk(tree):
                if isinstance(node, ast.Lambda):
                    raise SyntaxError("Starlark does not support lambda expressions.")

        # Compilation
        # compile_restricted needs the source string, not AST for eval mode
        self.bytecode = compile_restricted(self.code_string, filename='<string>', mode=self.mode)

    def _prepare_scope(self, global_vars: Dict[str, Any], global_funcs: Dict[str, Callable]) -> Dict[str, Any]:
        """Merges inputs into a safe execution scope."""

        sanitized_vars = {k: _sanitize_data(v) for k, v in global_vars.items()}

        scope = {
            '__builtins__': self.builtins,
            '_getattr_': _starlark_getattr,
            '_getitem_': lambda obj, index: obj[index],
            '_getiter_': lambda obj: iter(obj),
            '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
            '_unpack_sequence_': guarded_unpack_sequence,
            '_write_': lambda obj: obj, # Basic write guard
            '_print_': lambda *args, **kwargs: None, # Disable print
        }
        scope.update(sanitized_vars)
        scope.update(global_funcs)

        return scope

# --- StarlarkEval (Expressions) ---

class StarlarkEval(StarlarkBase):
    def __init__(self, expression: str):
        super().__init__(expression, mode='eval')

    def run(self, global_vars: Dict[str, Any], global_funcs: Dict[str, Callable]) -> Any:
        scope = self._prepare_scope(global_vars, global_funcs)
        try:
            return eval(self.bytecode, scope)
        except Exception as e:
            raise ToolException(f"Failed to evaluate ${{{self.code_string}}}: {e}")

# --- StarlarkExec (Scripts) ---

class StarlarkExec(StarlarkBase):
    def __init__(self, script: str):
        super().__init__(script, mode='exec')

    def run(self, global_vars: Dict[str, Any], global_funcs: Dict[str, Callable]) -> Any:
        scope = self._prepare_scope(global_vars, global_funcs)

        # Execute the script
        exec(self.bytecode, scope)

        # Determine Result Strategy
        # 1. Look for 'result' variable
        if 'result' in scope:
            return scope['result']

        # 2. Look for 'run()' function
        elif 'run' in scope and callable(scope['run']):
            # 'run' is a RestrictedPython function, so we call it safely
            return scope['run']()

        else:
            raise RuntimeError("Script must either define a 'result' variable or a 'run()' function.")