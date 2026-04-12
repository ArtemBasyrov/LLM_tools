"""
Tool-calling chat loop for a local Ollama model.

Run:
    micromamba run -n internet python chat.py
"""

import textwrap

import ollama
import tools.builtins  # noqa: F401  — registers built-in tools
import tools.web  # noqa: F401  — registers web_search and read_file
from tools import call, schemas

MODEL = "qwen3.5:35b"

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When answering questions that involve time-sensitive or current information "
    "(e.g. 'right now', 'today', 'currently', 'latest', 'recent'), "
    "always call get_current_datetime FIRST to establish the current date and time "
    "before performing any web search. "
    "Include the current date in your search query so results are temporally anchored."
)

# ANSI colours — gracefully degrade if the terminal doesn't support them
_DIM = "\033[2m"
_RESET = "\033[0m"
_THINKING_WIDTH = 80


def _print_thinking(text: str) -> None:
    """Print the model's reasoning block, dimmed and wrapped."""
    if not text or not text.strip():
        return
    print(f"{_DIM}┌─ thinking {'─' * (_THINKING_WIDTH - 12)}┐")
    for line in text.strip().splitlines():
        for wrapped in textwrap.wrap(line, width=_THINKING_WIDTH - 4) or [""]:
            print(f"│ {wrapped}")
    print(f"└{'─' * (_THINKING_WIDTH - 1)}┘{_RESET}")


def chat() -> None:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    print(f"Chatting with {MODEL}  (Ctrl-C or type 'exit' to quit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("exit", "quit", ""):
            print("Bye.")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop — keep going until the model returns plain text
        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=schemas(),
                think=True,
            )
            msg = response.message

            # Show reasoning if the model produced any
            _print_thinking(getattr(msg, "thinking", None))

            # Append assistant turn to history
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls or [],
                }
            )

            if not msg.tool_calls:
                # Model is done — print final answer
                print(f"\nAssistant: {msg.content}\n")
                break

            # Execute each requested tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments  # already a dict
                print(f"  [tool call] {fn_name}({fn_args})")

                result = call(fn_name, fn_args)
                print(f"  [tool result] {result}")

                messages.append({"role": "tool", "content": str(result)})


if __name__ == "__main__":
    chat()
