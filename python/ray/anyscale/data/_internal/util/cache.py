from typing import Callable, Dict, Tuple, Any
from contextlib import contextmanager
import time
import functools

_IS_TIMED_CACHE_ENABLED = True


def enable_timed_cache():
    global _IS_TIMED_CACHE_ENABLED
    _IS_TIMED_CACHE_ENABLED = True


def disable_timed_cache():
    global _IS_TIMED_CACHE_ENABLED
    _IS_TIMED_CACHE_ENABLED = False


# This is for testing purposes
@contextmanager
def _disable_timed_cache_for_tests():
    disable_timed_cache()
    yield
    enable_timed_cache()


def timed_cache(ttl: float, get_time_fn: Callable[[], float] = time.time):
    """
    Decorator that caches function results for a given TTL (in seconds).

    Args:
        ttl: Time-to-live in seconds for each cache entry.
    """

    def decorator(fn: Callable):
        cache: Dict[Tuple, Tuple[float, Any]] = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _IS_TIMED_CACHE_ENABLED:
                return fn(*args, **kwargs)
            key = args + tuple(sorted(kwargs.items()))
            try:
                hash(key)
            except TypeError as e:
                raise ValueError(
                    f"'`timed_cache` only supports arguments that are hashable, but '{key}' isn't hashable"
                ) from e
            now = get_time_fn()
            if key in cache:
                cached_time, value = cache[key]
                if now - cached_time < ttl:
                    return value  # Cache hit

            result = fn(*args, **kwargs)
            cache[key] = (now, result)
            return result

        return wrapper

    return decorator
