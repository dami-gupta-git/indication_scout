"""W&B lifecycle utilities."""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

import wandb

logger = logging.getLogger(__name__)


def wandb_run(project: str, tracked_param: str = "drug_name") -> Callable:
    """Decorator that wraps an async function in a W&B run.

    Reads the value of `tracked_param` from the wrapped function's arguments and
    passes it as run config. Calls wandb.finish() after the function returns,
    even if it raises.

    Args:
        project: W&B project name.
        tracked_param: Name of the parameter in the wrapped function that holds
            the drug name (default: "drug_name").
    """

    # Outer function receives the decorator arguments (project name, param name).
    def decorator(fn: Callable) -> Callable:

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:

            # Bind the call's positional and keyword arguments to the function's
            # parameter names, then fill in any defaults that weren't passed.
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            param_value = bound.arguments.get(tracked_param)

            # Start a new W&B run. reinit=True allows calling this more than once
            # in the same process
            wandb.init(
                project=project,
                config={tracked_param: param_value},
                reinit=True,
            )

            # Use try/finally so wandb.finish() is always called
            try:
                return await fn(*args, **kwargs)
            finally:
                wandb.finish()

        return wrapper

    return decorator
