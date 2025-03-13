import os
import logging
import hashlib
import subprocess
import sys
from typing import Callable, Any, List, Union

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


def setup_logging(console_level: int = logging.INFO, file_level: int = logging.DEBUG) -> None:
    """Configures logging to console and file"""
    logging.basicConfig(
        filename="llm-workers.log",
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
