import functools
import hashlib
import inspect
import logging
import os

from diskcache import Cache

from bpm_ai_core.llm.common.blob import Blob

logger = logging.getLogger(__name__)


_cache = Cache(directory=os.path.join(os.path.expanduser("~"), ".cache", "bpm-ai", "predictions"))


def calculate_cache_key(data: tuple) -> str:
    hash_input = repr(data).encode('utf-8')
    cache_key = hashlib.sha256(hash_input).hexdigest()
    return cache_key


def cached(exclude: list[str] = None, key_func=None, disable_if: str = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if disable_if:
                sig = inspect.signature(func)
                # Bind the passed arguments to the parameters in the signature partially
                bound_args = sig.bind_partial(self, *args, **kwargs)
                bound_args.apply_defaults()
                # Create a dictionary of parameter names to values
                param_dict = bound_args.arguments

                if eval(disable_if, {"self": self, **param_dict}):
                    # If the expression is true, bypass caching and call the original function
                    return await func(self, *args, **kwargs)

            cache_key_components = []

            # include all (non-excluded) constructor parameters
            if hasattr(self, "__constructor_params__"):
                cache_key_components.extend(f"{k}={v}" for k, v in sorted(self.__constructor_params__.items()))
            else:
                logger.warning("Skipping cache: __constructor_params__ not set, decorate class with @cachable()")
                # bypass caching and call the original function
                return await func(self, *args, **kwargs)

            # Include all function parameters, excluding the specified ones
            exclude_params = set(exclude) if exclude else set()
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(self, *args, **kwargs)
            bound_args.apply_defaults()
            for param_name, param_value in bound_args.arguments.items():
                if param_name not in exclude_params and param_name != "self":
                    cache_key_components.append(f"{param_name}={param_value}")

            if key_func:
                custom_key_part = await key_func(*args, **kwargs)
                cache_key_components.append(custom_key_part)

            logger.debug(f"Cache key components: {cache_key_components}")

            cache_key = hashlib.sha256(":".join(cache_key_components).encode()).hexdigest()
            cache_key = f"{self.__class__.__module__}__{self.__class__.__qualname__}__{func.__name__}__" + cache_key

            cached_result = _cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for key '{cache_key}': {cached_result}")
                return cached_result

            result = await func(self, *args, **kwargs)
            _cache.set(cache_key, result)
            return result

        return wrapper

    return decorator


def cachable(exclude_key_params=None):
    if exclude_key_params is None:
        exclude_key_params = []

    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            # Get the constructor parameters and their values
            constructor_params = inspect.signature(original_init).parameters
            constructor_values = {
                param: value
                for param, value in zip(constructor_params, args)
                if param not in exclude_key_params
            }
            constructor_values.update({
                param: value
                for param, value in kwargs.items()
                if param not in exclude_key_params
            })

            # Save the constructor parameters and values in a special instance variable
            self.__constructor_params__ = constructor_values

            # Call the original constructor
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__

        return cls

    return decorator


async def blob_cache_key(blob_or_path: Blob | str, *args, **kwargs) -> str:
    if isinstance(blob_or_path, str):
        blob = Blob.from_path_or_url(blob_or_path)
    else:  # Blob
        blob = blob_or_path
    return f"blob_bytes={await blob.as_bytes()}"
