"""
Tool registry.

Usage:
    from tools import register, schemas, call

    @register(
        description="What this tool does.",
        parameters={
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "..."},
            },
            "required": ["arg"],
        },
    )
    def my_tool(arg: str) -> str:
        return arg.upper()
"""

from typing import Any, Callable

_registry: dict[str, dict] = {}  # name -> {schema, fn}


def register(description: str, parameters: dict | None = None):
    """Decorator that registers a function as a callable tool."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "required": []}

    def decorator(fn: Callable) -> Callable:
        _registry[fn.__name__] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": parameters,
                },
            },
            "fn": fn,
        }
        return fn

    return decorator


def schemas() -> list[dict]:
    """Return all registered tool schemas (pass directly to ollama)."""
    return [entry["schema"] for entry in _registry.values()]


def call(name: str, arguments: dict) -> Any:
    """Execute a registered tool by name with the given arguments."""
    entry = _registry.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'"
    try:
        return entry["fn"](**arguments)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
