import hashlib
import logging
import os
import time
from typing import Any
from typing import Callable, List, Optional

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

