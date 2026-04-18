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

_registry: dict[str, dict] = {}  # name -> {schema, fn, always_on}
_active_tools: set[str] = set()  # tools activated via load_tools for this session


def register(description: str, parameters: dict | None = None, always_on: bool = False):
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
            "always_on": always_on,
        }
        return fn

    return decorator


def activate(name: str) -> bool:
    """Add a tool to the active set. Returns False if tool not found."""
    if name not in _registry:
        return False
    _active_tools.add(name)
    return True


def schemas() -> list[dict]:
    """Return schemas for always-on tools plus any activated tools."""
    visible = {n for n, e in _registry.items() if e["always_on"]} | _active_tools
    return [e["schema"] for n, e in _registry.items() if n in visible]


def call(name: str, arguments: dict) -> Any:
    """Execute a registered tool by name with the given arguments."""
    entry = _registry.get(name)
    if entry is None:
        return f"Error: unknown tool '{name}'"
    try:
        return entry["fn"](**arguments)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
