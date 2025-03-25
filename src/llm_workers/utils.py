import fnmatch
import hashlib
import logging
import mimetypes
import platform
import subprocess
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Callable, Any, List, Optional

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import ToolCall, BaseMessage
from langchain_core.tools import ToolException

logger =  logging.getLogger(__name__)


def _build_cache_filename(source_file_paths: List[str], cache_file_suffix: str, discriminator: str) -> str:
    md5 = hashlib.md5()
    for source_file_path in source_file_paths:
        md5.update(source_file_path.encode())
    if discriminator is not None:
        md5.update(discriminator.encode())
    filename = md5.hexdigest()
    return f"{filename}{cache_file_suffix}"


def cached(
        input_path: str,
        cache_file_suffix: str,
        func: Callable[[str], Any],
        discriminator: str = None
) -> str:
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
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)

    cached_filename = _build_cache_filename(input_paths, cache_file_suffix, discriminator)
    cached_path = os.path.join(cache_dir, cached_filename)

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
        logger.debug(f"Cached file {cached_filename} is up-to-date")
        return cached_path

    try:
        func(cached_path)
        return cached_path
    except Exception:
        logger.info(f"Deleted cached file {cached_filename} due to error")
        if os.path.exists(cached_path):
            os.remove(cached_path)
        raise


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


def get_environment_variable(name: str, default: str | None) -> str | None:
    return os.environ.get(name, default)

def ensure_environment_variable(name: str) -> str:
    var = os.environ.get(name)
    if var is None:
        raise ToolException(f"Environment variable {name} not set")
    return var


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


def setup_logging(args: Namespace, log_filename: Optional[str] = None) -> None:
    """Configures logging to console and file in a standard way.
    Args:
        args: command line arguments to look for `--verbose` and `--debug`
        log_filename: (optional) name of the log file, if not specified name will be derived from script name
    """
    if log_filename is None:
        log_filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + ".log"

    console_level: int = logging.WARNING
    file_level: int = logging.INFO
    if args.verbose:
        console_level = logging.INFO
    if args.debug:
        file_level = logging.DEBUG

    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
        level=file_level
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


class LazyFormatter:
    def __init__(self, target, custom_formatter: Callable[[Any], str] = None):
        self.target = target
        self.custom_formatter = custom_formatter
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
            elif isinstance(self.target, BaseMessage):
                self.repr = self.target.pretty_repr()
            else:
                self.repr = repr(self.target)
        return self.repr


def find_and_load_dotenv(path_from_home_dir: str):
    """Tries to find and load .env file. Order:
    1. Current directory
    2. Parent directories of current directory
    3. Home directory

    Args:
        path_from_home_dir: path of the file within home directory
    """
    env_path = None
    # 1. check current directory and parent directories
    std_env_path = find_dotenv(usecwd=True)
    if std_env_path and os.path.exists(std_env_path):
        env_path = std_env_path

    # 2. check path within home directory
    if not env_path:
        home_dir = os.path.expanduser("~")
        path = os.path.join(home_dir, path_from_home_dir)
        if os.path.exists(path):
            env_path = path

    if env_path:
        logger.info(f"Loading {env_path}")
        return load_dotenv(env_path)
    return False



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
        logger.warning(f"Failed to open file {filepath} in default app: {e}", exc_info=True)
        return False