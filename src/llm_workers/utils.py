import fnmatch
import hashlib
import logging
import mimetypes
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Any, List, Optional

import yaml
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import ToolCall
from pydantic import BaseModel

logger =  logging.getLogger(__name__)

####################################################
# Cache
####################################################

# Global variable to track when we last cleaned the cache
_last_cache_cleaning_time = 0
# Cache configuration/state
_cache_dir: str = '.cache'
_cache_ttl: Optional[int] = None  # also serves as "prepared" flag
_cache_warning_emitted: bool = False

# Environment variable configuration/state
_env_file_path: Optional[str] = None  # tracks which .env file was loaded or would be used


def _ensure_cache_dir_exists(create: bool = True) -> str:
    """Ensure cache directory exists and return its path.
    If create is False, do not create the directory if missing.
    """
    if create and not os.path.exists(_cache_dir):
        os.makedirs(_cache_dir, exist_ok=True)
    return _cache_dir


_cache_cleanup_interval_sec = 300  # 5 minutes

def is_cache_prepared() -> bool:
    """Check if cache is prepared (i.e., prepare_cache was called)."""
    return _cache_ttl is not None

def _prepare_cache(create_dir: bool, now: Optional[float] = None) -> str:
    """Cleanup expired cache files if prepared; otherwise warn once.
    Does not create the cache dir; if not present, silently returns.
    """
    global _cache_dir, _last_cache_cleaning_time, _cache_warning_emitted, _cache_cleanup_interval_sec

    dir_exists = os.path.exists(_cache_dir)
    if not dir_exists and create_dir:
        os.makedirs(_cache_dir, exist_ok=True)
        dir_exists = True

    if not is_cache_prepared():
        if not _cache_warning_emitted:
            logger.warning("prepare_cache not called, cache will not be cleaned")
            _cache_warning_emitted = True
        return _cache_dir

    if now is None:
        now = time.time()

    if now - _last_cache_cleaning_time <= _cache_cleanup_interval_sec:
        return _cache_dir

    if not dir_exists:
        return _cache_dir

    removed_files = 0
    for file in os.listdir(_cache_dir):
        file_path = os.path.join(_cache_dir, file)
        try:
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path) > _cache_ttl):
                logger.debug('Removing expired cache file %s', file_path)
                os.remove(file_path)
                removed_files += 1
        except FileNotFoundError:
            # File might be removed concurrently; ignore
            pass
        except Exception:
            logger.debug("Failed to check/remove cache file %s", file_path, exc_info=True)
    if removed_files > 0:
        logger.info(f"Removed {removed_files} expired cache files from {_cache_dir}")

    _last_cache_cleaning_time = now
    return _cache_dir

def prepare_cache(ttl: int = 3600*24) -> str:
    """Prepare cache by setting TTL, optionally creating directory, and cleaning up old files.
    Args:
        ttl: time to live for cache files in seconds
    Returns:
        path to cache directory
    """
    global _cache_ttl

    if _cache_ttl is not None:
        logger.debug(f"prepare_cache already called, ignoring subsequent call with ttl={ttl}")
        return _cache_dir

    _cache_ttl = ttl
    logger.debug(f"Cache ttl is set to {ttl} seconds")
    return _prepare_cache(create_dir=False)


def get_cache_dir() -> str:
    """Return cache directory path, ensuring it exists and attempting cleanup.
    If prepare_cache() wasn't called before, a warning will be logged (once).
    """
    return _prepare_cache(create_dir=True)


def get_cache_filename(input: any, suffix: str) -> str:
    """Return content-based cache filename with given suffix.
    Triggers cleanup which will warn once if prepare_cache() wasn't called.
    """
    return os.path.join(get_cache_dir(), f"{calculate_hash(input)}{suffix}")


def calculate_hash(input: any) -> str:
    """Calculate MD5 hash of given input."""
    hasher = hashlib.md5()
    _hash_any(input, hasher)
    return hasher.hexdigest()

def _hash_any(value: any, hasher: hashlib.md5) -> None:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            hasher.update(str(key).encode())
            _hash_any(value[key], hasher)
    elif isinstance(value, list):
        for item in value:
            _hash_any(item, hasher)
    else:
        hasher.update(str(value).encode())

def cached(
        input_path: str,
        cache_file_suffix: str,
        func: Callable[[str], Any],
        discriminator: str = None
) -> str:
    """Calculates cache file path, and calls provided function only if the cache file is older than the input file.

    Args:
        input_path: paths to the input file
        cache_file_suffix: suffix for file name in cache, usually extension like `.wav`
        func: function to call if the cache file doesn't exist or is older than the input file. The sole input
        argument to this function is the absolute path to the cache file.
        discriminator: if specified, md5 hash of it is appended to cache filename to differentiate between different
        parameters used in transformation process.
    """
    return multi_cached([input_path], cache_file_suffix, func, discriminator)

def multi_cached(
        input_paths: List[str],
        cache_file_suffix: str,
        func: Callable[[str], Any],
        discriminator: str = None
) -> str:
    """Calculates cache file path, and calls provided function only if the cache file is older than the input files.

    Args:
        input_paths: paths to the input files
        cache_file_suffix: suffix for file name in cache, usually extension like `.wav`
        func: function to call if the cache file doesn't exist or is older than the input file. The sole input
        argument to this function is the absolute path to the cache file.
        discriminator: if specified, md5 hash of it is appended to cache filename to differentiate between different
        parameters used in transformation process.
    """
    cached_path = get_cache_filename([input_paths, discriminator], cache_file_suffix)

    needs_run = False
    if not os.path.exists(cached_path):
        logger.debug(f"{cached_path} not found, recomputing...")
        needs_run = True
    else:
        for input_path in input_paths:
            if os.path.getmtime(cached_path) < os.path.getmtime(input_path):
                logger.debug(f"{cached_path} not found or is older than {input_path}, recomputing...")
                needs_run = True
                break
    if not needs_run:
        logger.debug(f"Cached file {cached_path} is up-to-date")
        return cached_path

    try:
        func(cached_path)
        return cached_path
    except Exception:
        logger.info(f"Deleted cached file {cached_path} due to error")
        if os.path.exists(cached_path):
            try:
                os.remove(cached_path)
            except Exception:
                logger.debug("Failed to remove errored cache file %s", cached_path, exc_info=True)
        raise


####################################################
# Execution Environment
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

def ensure_environment_variable(var_name: str, description: str = None) -> str:
    """
    Ensure an environment variable is set, prompting the user if it's missing.

    Args:
        var_name: Name of the environment variable
        description: Optional description to show to the user

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
        return value

    # Variable is not set, prompt the user
    print(f"\nEnvironment variable '{var_name}' is not set.")
    if description:
        print(f"This is: {description}")
    print(f"\nThe value will be saved to: {_env_file_path}")
    print("If you don't want this, exit with Ctrl-C and run the program with:")
    print(f"  {var_name}=your_token {Path(sys.argv[0]).name}\n")

    # Get input from user
    try:
        value = input(f"Enter value for {var_name}: ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(1)

    # Validate input
    if not value:
        print(f"Error: {var_name} cannot be empty")
        exit(1)

    # Try to save to .env file
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
    console_level: int = logging.FATAL
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


####################################################
# Misc.
####################################################

def format_tool_call(tc: ToolCall) -> str:
    name = tc.get('name', '<tool>')
    args = tc.get("args")
    return format_tool_invocation(name, args)

def format_tool_invocation(name: str, args: Any) -> str:
    if isinstance(args, dict):
        arg = next(iter(args.values()), None)
        if arg is None:
            return name
        else:
            args = str(arg)
    else:
        args = str(args)
    limit = 80
    if len(args) > limit:
        return f"{name} \"{args[:limit]}...\""
    else:
        return f"{name} \"{args}\""

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
    except Exception as e:
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
