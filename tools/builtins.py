"""
Built-in demo tools — replace or extend with your own.
"""

import datetime
from tools import register


@register(
    description=(
        "Return the current local date and time. "
        "Call this whenever: (1) the user's question involves 'now', 'today', 'current', "
        "'latest', 'recent', or any other time-relative term — before doing a web search — "
        "so you know exactly what date to anchor the query to; "
        "(2) you are about to write a file that includes a timestamp, date, or version "
        "number in its name or content. "
        "Examples: "
        "user asks 'What happened today?' → call get_current_datetime first, then web_search with the date; "
        "user says 'save a log with today's date in the filename' → call get_current_datetime to get the date string."
    ),
    always_on=True,
)
def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@register(
    description=(
        "Evaluate a safe arithmetic expression and return the result. "
        "Use this for any arithmetic calculation to avoid hallucinating numbers. "
        "Supports +, -, *, /, ** (power), and parentheses only — no functions or variables. "
        "Examples: "
        "'2 ** 10 + 5 / 2' → 1026.5; "
        "'(100 - 37) * 1.15' → 72.45; "
        "'1234567 * 9876543' → 12193254061481. "
        "NOT for: string operations, date math, or expressions with variables."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression using digits, +, -, *, /, **, and parentheses. e.g. '(512 * 1024) / 8'",
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
