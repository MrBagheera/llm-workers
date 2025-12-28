import fnmatch
import importlib.resources
import json
import logging
import mimetypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Callable, List, Optional, Dict

import yaml
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

logger =  logging.getLogger(__name__)


####################################################
# Execution Environment
####################################################

def find_and_load_dotenv(fallback_path: Path):
    """Tries to find and load .env file. Order:
    1. Current directory
    2. Parent directories of current directory
    3. Fallback path

    Args:
        fallback_path: path of the file within home directory
    """
    global _env_file_path

    env_path = None
    # 1. check current directory and parent directories
    std_env_path = find_dotenv(usecwd=True)
    if std_env_path and os.path.exists(std_env_path):
        env_path = std_env_path

    # 2. check fallback path
    if not env_path:
        if os.path.exists(fallback_path):
            env_path = fallback_path

    # Always set the env file path, even if no file was found (use fallback path)
    _env_file_path = env_path if env_path else str(fallback_path)

    if env_path:
        logger.info(f"Loading {env_path}")
        return load_dotenv(env_path)
    return False

def get_env_var_or_fail(name: str) -> str:
    var = os.environ.get(name)
    if var is None:
        raise OSError(f"Environment variable {name} not set")
    return var

def ensure_environment_variable(environment: Dict[str,str], var_name: str, description: any = None, is_persistent: bool = True, is_secret: bool = False) -> str:
    """
    Ensure an environment variable is set, prompting the user if it's missing.

    Args:
        environment: Dictionary representing the current environment
        var_name: Name of the environment variable
        description: Optional description to show to the user
        is_persistent: If True, save to .env file; if False, only set for current session
        is_secret: Should input be replaced by asterisks (if possible)

    Returns:
        The value of the environment variable

    Raises:
        RuntimeError: If find_and_load_dotenv was not called prior to this function
    """
    global _env_file_path


    if _env_file_path is None:
        raise RuntimeError("find_and_load_dotenv must be called before ensure_environment_variable")

    # Check if the variable is already set
    value = os.environ.get(var_name)
    if value is not None:
        environment[var_name] = value
        return value

    # Variable is not set, prompt the user
    if description:
        from llm_workers.expressions import StringExpression
        if isinstance(description, StringExpression):
            description = description.evaluate({'env': environment})
        print(f"\nPlease provide value for '{var_name}': {description}")
    else:
        print(f"\nPlease provide value for '{var_name}'.")
    if is_persistent:
        print(f"The value will be saved to: {_env_file_path}")
        print("If you don't want this, exit with Ctrl-C and run the program with:")
        print(f"  {var_name}=your_token {Path(sys.argv[0]).name}")
    else:
        print("This variable will be used, but not saved to disk.")

    # Get input from user
    try:
        if is_secret and sys.stdin.isatty():
            from prompt_toolkit import prompt
            value = prompt('Value (input hidden): ', is_password=True)
        else:
            value = input("Value: ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(1)

    # Validate input
    if not value:
        print(f"Error: {var_name} cannot be empty")
        exit(1)

    # Try to save to .env file if persistent
    if is_persistent:
        try:
            env_path = Path(_env_file_path)
            # Ensure the directory exists
            env_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to the .env file
            with open(env_path, 'a', encoding='utf-8') as f:
                f.write(f"{var_name}={value}\n")

            print(f"Successfully saved {var_name} to {env_path}")
        except Exception as e:
            logger.warning(f"Failed to save {var_name} to {_env_file_path}: {e}")
            print(f"Warning: Could not save to {_env_file_path}, but continuing with entered value")

    # Set the environment variable for this session
    os.environ[var_name] = value
    environment[var_name] = value

    return value


####################################################
# Logging
####################################################

DEBUG_LOGGERS = (
    ["llm_workers.worker"],
    ["llm_workers"],
)

def setup_logging(
        debug_level: int,
        debug_loggers_by_debug_level: list[list[str]] = DEBUG_LOGGERS,
        verbosity: int = 0,
        log_filename: Optional[str] = None
) -> str:
    """Configures logging to console and file in a standard way.
    Args:
        debug_level: verbosity level for file logging
        debug_loggers_by_debug_level: list of debug loggers by debug level
        verbosity: verbosity level for console logging (0 - ERROR & WARN, 1 - INFO, 2 - DEBUG)
        log_filename: (optional) name of the log file, if not specified name will be derived from script name
    """
    # file logging
    if log_filename is None:
        log_filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".log"
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    # adjust levels for individual loggers at given debug level
    if debug_level >= len(debug_loggers_by_debug_level):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        for logger_name in debug_loggers_by_debug_level[debug_level]:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)

    # console logging
    console_level: int = logging.ERROR
    if verbosity == 1:
        console_level = logging.INFO
    elif verbosity >= 2:
        console_level = logging.DEBUG
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    formatter = logging.Formatter("%(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    return os.path.abspath(log_filename)

def format_as_yaml(obj: Any, trim: bool) -> str:
    """Format given object as YAML string with optional trimming of all string fields recursively.

    Args:
        obj: object to format
        trim: If True, trims string fields longer than 80 characters and truncates multiline strings to the first line.

    Returns:
        A YAML-formatted string representation of the messages
    """
    raw = _to_json_compatible(obj)

    if trim:
        raw = _trim_recursively(raw)

    return yaml.dump(raw, default_flow_style=False, sort_keys=False, allow_unicode=True)

def _to_json_compatible(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_compatible(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _to_json_compatible(value) for key, value in obj.items()}
    if hasattr(obj, '__dict__'):
        return _to_json_compatible(vars(obj))
    return repr(obj)

def _trim_recursively(data):
    if isinstance(data, dict):
        return {key: _trim_recursively(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_trim_recursively(item) for item in data]
    elif isinstance(data, str):
        lines = data.splitlines()
        if len(lines) > 0:
            line = lines[0]
            return line[:77] + "..." if len(line) > 80 or len(lines) > 1 else line
    return data

class LazyFormatter:
    def __init__(self, target, custom_formatter: Callable[[Any], str] = None, trim: bool = True):
        self.target = target
        self.custom_formatter = custom_formatter
        self.trim = trim
        self.repr = None
        self.str = None

    def __str__(self):
        if self.str is None:
            if self.custom_formatter is not None:
                self.str = self.custom_formatter(self.target)
                self.repr = self.str
            else:
                self.str = str(self.target)
        return self.str

    def __repr__(self):
        if self.repr is None:
            if self.custom_formatter is not None:
                self.str = self.custom_formatter(self.target)
                self.repr = self.str
            else:
                self.repr = format_as_yaml(self.target, self.trim)
        return self.repr


## YAML Loader with !include and !require

class SmartLoader(yaml.SafeLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.getcwd()
        super().__init__(stream)

    def _load_file(self, filename):
        """
        Helper method that handles path resolution and
        smart parsing (JSON/YAML vs Text).
        """
        # validate file does not escape current directory
        if ".." in filename.split(os.path.sep):
            raise ValueError(f"Relative paths cannot escape current directory: {filename}")
        if filename.startswith("./"):
            # Relative to current directory
            filepath = os.path.join(os.getcwd(), filename[2:])
        else:
            # Relative to the YAML file directory
            filepath = os.path.join(self._root, filename)

        # Check extensions to decide how to load
        extension = os.path.splitext(filename)[1].lower()

        with open(filepath, 'r') as f:
            if extension in ['.yaml', '.yml']:
                return yaml.load(f, Loader=SmartLoader)
            elif extension == '.json':
                return json.load(f)
            else:
                return f.read()

    def include(self, node):
        """
        Implementation of !include
        Returns empty string if file is missing.
        """
        filename = self.construct_scalar(node)
        try:
            return self._load_file(filename)
        except FileNotFoundError:
            # Graceful failure: return empty string
            return ""

    def require(self, node):
        """
        Implementation of !require
        Raises error if file is missing.
        """
        filename = self.construct_scalar(node)
        # This will naturally raise FileNotFoundError if missing,
        # which stops the loading process.
        return self._load_file(filename)

# Register both tags
SmartLoader.add_constructor('!include', SmartLoader.include)
SmartLoader.add_constructor('!require', SmartLoader.require)

def load_yaml(file_path: str | Path) -> Any:
    """Load YAML file using SmartLoader with !include and !require support.
    Also supports loading from "module:file" syntax.

    Args:
        file_path: path to the YAML file"""
    if isinstance(file_path, str) and ':' in file_path:
        module, resource = file_path.split(':', 1)
        if len(module) > 1: # ignore volume names on windows
            # noinspection PyUnresolvedReferences
            with importlib.resources.files(module).joinpath(resource).open("r") as file:
                return yaml.load(file, Loader=SmartLoader)
    # default - local file loading
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=SmartLoader)


####################################################
# Misc.
####################################################

class RunProcessException(IOError):
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause

def run_process(cmd: List[str]) -> str:
    cmd_str = LazyFormatter(cmd, custom_formatter = lambda x: " ".join(x))
    logger.debug("Running %s", cmd_str)
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        (result, stderr_data) = process.communicate()
        exit_code = process.wait()
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise RunProcessException(f"Running sub-process [{cmd_str}] failed with error: {e}", e)
    if exit_code == 0:
        logger.debug("Sub-process [%s] finished with exit code %s, result_len=%s, stderr:\n%s", cmd_str, exit_code, len(result), stderr_data)
        return result
    else:
        raise RunProcessException(f"Sub-process [{cmd_str}] finished with exit code {exit_code}, result_len={len(result)}, stderr:\n{stderr_data}")


class FileChangeDetector:
    def __init__(self, path: str, included_patterns: list[str], excluded_patterns: list[str]):
        self.path = path
        self.included_patterns = included_patterns
        self.excluded_patterns = excluded_patterns
        self.last_snapshot = self._snapshot()

    def _should_include(self, filename):
        included = any(fnmatch.fnmatch(filename, pattern) for pattern in self.included_patterns)
        if not included:
            return False
        excluded = any(fnmatch.fnmatch(filename, pattern) for pattern in self.excluded_patterns)
        return not excluded

    def _snapshot(self):
        """Take a snapshot of all non-ignored files and their modification times."""
        return {
            f: os.path.getmtime(os.path.join(self.path, f))
            for f in os.listdir(self.path)
            if os.path.isfile(os.path.join(self.path, f)) and self._should_include(f)
        }

    def check_changes(self):
        """Compare current snapshot to previous, and return changes."""
        current_snapshot = self._snapshot()

        created = [f for f in current_snapshot if f not in self.last_snapshot]
        deleted = [f for f in self.last_snapshot if f not in current_snapshot]
        modified = [
            f for f in current_snapshot
            if f in self.last_snapshot and current_snapshot[f] != self.last_snapshot[f]
        ]

        self.last_snapshot = current_snapshot
        return {'created': created, 'deleted': deleted, 'modified': modified}


DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.ps1',
    '.sh', '.bash', '.zsh', '.py', '.pyw', '.pl', '.rb',
    '.app', '.desktop', '.jar', '.msi', '.vb', '.wsf'
}

def is_safe_to_open(filepath: Path | str) -> bool:
    if not isinstance(filepath, Path):
        filepath = Path(str(filepath))
    ext = filepath.suffix.lower()
    if ext in DANGEROUS_EXTENSIONS:
        return False

    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        if mime_type.startswith('application/x-executable') or \
                mime_type.startswith('application/x-msdownload') or \
                mime_type.startswith('application/x-sh'):
            return False
    return True

def open_file_in_default_app(filepath: str) -> bool:
    path = Path(filepath)
    if not path.exists():
        logger.warning(f"Cannot open file {filepath} in default app: file does not exist")
        return False

    if not is_safe_to_open(path):
        logger.warning(f"Blocked potentially dangerous file {filepath} from opening in default app")
        return False

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(path)
        elif system == 'Darwin':
            subprocess.run(['open', str(path)])
        else:
            subprocess.run(['xdg-open', str(path)])
        return True
    except Exception:
        logger.warning(f"Failed to open file {filepath} in default app", exc_info=True)
        return False


def get_key_press() -> str:
    """Get a single key press from the user without requiring Enter.
    
    Returns:
        The pressed key as a string
    """
    if sys.platform == 'win32':
        import msvcrt
        key = msvcrt.getch()
        return key.decode('utf-8', errors='ignore')
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key


def _split_type_parameters(s: str) -> list[str]:
    """Split type parameters by comma, respecting nested brackets.
    
    Example: "str, dict[str, int]" -> ["str", "dict[str, int]"]
    """
    parts = []
    current_part = ""
    bracket_depth = 0
    
    for char in s:
        if char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == ',' and bracket_depth == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        
        current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    return parts

def parse_standard_type(s: str):
    if s == "str":
        return str
    elif s == "int":
        return int
    elif s == "float":
        return float
    elif s == "bool":
        return bool
    elif s == "dict":
        return dict
    elif s == "list":
        return list
    elif s.startswith("literal:"):
        # Extract the values after "literal:" and split by "|"
        literal_values = s[len("literal:"):].split("|")
        from typing import Literal
        # Create a Literal type with the extracted values
        return Literal[tuple(literal_values)]
    elif s.startswith("list[") and s.endswith("]"):
        # Handle parametrized lists like "list[str]", "list[int]"
        inner_type_str = s[5:-1]  # Remove "list[" and "]"
        inner_type = parse_standard_type(inner_type_str)
        from typing import List
        return List[inner_type]
    elif s.startswith("dict[") and s.endswith("]"):
        # Handle parametrized dicts like "dict[str, int]", "dict[str, dict[str, int]]"
        inner_types_str = s[5:-1]  # Remove "dict[" and "]"
        type_parts = _split_type_parameters(inner_types_str)
        if len(type_parts) == 2:
            key_type = parse_standard_type(type_parts[0])
            value_type = parse_standard_type(type_parts[1])
            from typing import Dict
            return Dict[key_type, value_type]
        else:
            raise ValueError(f"Dict type must have exactly 2 parameters: {s}")
    else:
        raise ValueError(f"Unknown type: {s}")


def matches_patterns(tool_name: str, patterns: List[str]) -> bool:
    """
    Check if tool_name matches any of the patterns.
    Supports negation with ! prefix.

    Rules:
    - If pattern starts with !, it's a negation (exclude)
    - To match: must match at least one positive pattern AND not match any negative pattern
    - If only negative patterns exist, matches by default (like implicit "*")

    Examples:
        matches_patterns("gh_read", ["gh*", "!gh_write*"]) -> True
        matches_patterns("gh_write_file", ["gh*", "!gh_write*"]) -> False
        matches_patterns("any_tool", []) -> False
        matches_patterns("foo", ["!bar"]) -> True   # only exclusions, foo not excluded
        matches_patterns("bar", ["!bar"]) -> False  # only exclusions, bar is excluded
    """
    if not patterns:
        return False

    inclusions = [p for p in patterns if not p.startswith("!")]
    exclusions = [p[1:] for p in patterns if p.startswith("!")]

    # If only negative patterns, match by default (implicit "*")
    if not inclusions:
        included = True
    else:
        included = any(fnmatch.fnmatch(tool_name, pattern) for pattern in inclusions)

    # Apply exclusions
    if included and exclusions:
        excluded = any(fnmatch.fnmatch(tool_name, pattern) for pattern in exclusions)
        return not excluded

    return included
