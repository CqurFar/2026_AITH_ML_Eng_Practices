import functools
from collections.abc import Callable


def step(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[{name}] start", flush=True)
            result = func(*args, **kwargs)
            print(f"[{name}] done", flush=True)
            return result

        return wrapper

    return decorator
