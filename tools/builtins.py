"""
Built-in demo tools — replace or extend with your own.
"""

import datetime
from tools import register


@register(
    description=(
        "Return the current local date and time. "
        "Call this FIRST whenever the user's question involves 'now', 'today', 'current', "
        "'latest', 'recent', or any other time-relative term — before doing a web search — "
        "so you know exactly what date to anchor the query to."
    )
)
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@register(
    description="Evaluate a safe arithmetic expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression, e.g. '2 ** 10 + 5 / 2'",
            },
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: only basic arithmetic is allowed (+, -, *, /, **, parentheses)"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"
