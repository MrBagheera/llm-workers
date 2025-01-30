import os
import logging
import hashlib
import subprocess
from typing import Callable, Any, List

from langchain_core.tools import ToolException

logger =  logging.getLogger(__name__)


def _build_cache_filename(source_file_path: str, cache_file_suffix: str, discriminator: str) -> str:
    md5 = hashlib.md5()
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
    """Calculates cache file path, and calls provided function only if the cache file is older than the input file.

    Args:
        input_path: path to the input file
        cache_file_suffix: suffix for file name in cache, usually extension like `.wav`
        func: function to call if the cache file doesn't exist or is older than the input file. The sole input
        argument to this function is the absolute path to the cache file.
        discriminator: if specified, md5 hash of it is appended to cache filename to differentiate between different
        parameters used in transformation process.
    """
    cache_dir = ".cache"
    os.makedirs(cache_dir, exist_ok=True)

    cached_filename = _build_cache_filename(input_path, cache_file_suffix, discriminator)
    cached_path = os.path.join(cache_dir, cached_filename)

    if (not os.path.exists(cached_path)
            or os.path.getmtime(cached_path) < os.path.getmtime(input_path)):
        logger.debug(f"{cached_path} not found or is older than {input_path}, recomputing...")
        try:
            func(cached_path)
        except Exception:
            logger.info(f"Deleted cached file {cached_filename} due to error")
            if os.path.exists(cached_path):
                os.remove(cached_path)
            raise
    else:
        logger.debug(f"Cached file {cached_filename} is up-to-date")
    return cached_path


def run_process(cmd: List[str], stdout_transform: Callable[[str], Any] = None):
    cmd_str = " ".join(cmd)
    logger.debug(f"Running {cmd_str}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if stdout_transform is not None:
            result = []
            for line in process.stdout:
                transformed = stdout_transform(line)
                if transformed is not None:
                    result.append(transformed)
            stderr_data = process.stderr.read()  # Capture remaining stderr
        else:
            (result, stderr_data) = process.communicate()
        exit_code = process.wait()
        message = f"Sub-process [{cmd_str}] finished with exit code {exit_code}, result_len={len(result)}, stderr:\n{stderr_data}"
        if exit_code == 0:
            logger.debug(message)
            return result
        else:
            raise ToolException(message)
    except FileNotFoundError:
        raise ToolException("ffmpeg not found. Please install it.")
    except Exception as e:
        raise ToolException(f"Running sub-process [{cmd_str}] failed with error: {e}")

