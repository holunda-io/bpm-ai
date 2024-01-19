import inspect
from functools import wraps

from bpm_ai_core.tracing.tracing import Tracing


def trace(name: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = func.__name__ if name is None else name
            params = inspect.signature(func).parameters
            inputs = {**{list(params.keys())[i]: arg for i, arg in enumerate(args)}, **kwargs}
            Tracing.tracers().start_trace(trace_name, inputs)
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    outputs = result
                else:
                    outputs = {'output': result}
                Tracing.tracers().end_trace(outputs)
                return result
            except Exception as e:
                Tracing.tracers().end_trace({'error': str(e)})
                raise e
        return wrapper
    return decorator


def span(name: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = func.__name__ if name is None else name
            params = inspect.signature(func).parameters
            inputs = {**{list(params.keys())[i]: arg for i, arg in enumerate(args)}, **kwargs}
            Tracing.tracers().start_span(span_name, inputs)
            try:
                result = func(*args, **kwargs)
                if isinstance(result, dict):
                    outputs = result
                else:
                    outputs = {'output': result}
                Tracing.tracers().end_span(outputs)
                return result
            except Exception as e:
                Tracing.tracers().end_span({'error': str(e)})
                raise e
        return wrapper
    return decorator
