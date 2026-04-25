"""
Inference backend abstraction.

Default backend is ``llama_server`` (ik_llama.cpp's ``llama-server``), which
supports speculative decoding and is materially faster than Ollama on Apple
Silicon. Set ``LLM_BACKEND=ollama`` to fall back to Ollama (e.g. for
debugging or if llama-server is not running).

The module exposes ``chat(...)`` and ``show(...)`` with the same return
shape as the ``ollama`` Python SDK so existing call sites keep working
without changes.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Iterator

import httpx

_BACKEND = os.environ.get("LLM_BACKEND", "llama_server").strip().lower()
_LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")


# ---------------------------------------------------------------------------
# Ollama passthrough
# ---------------------------------------------------------------------------


def _ollama_chat(**kwargs):
    import ollama

    return ollama.chat(**kwargs)


def _ollama_show(model: str):
    import ollama

    return ollama.show(model)


# ---------------------------------------------------------------------------
# llama-server (OpenAI-compatible) backend
# ---------------------------------------------------------------------------


def _ns(d: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace so attribute access works
    (matching the Ollama SDK's response shape)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(x) for x in d]
    return d


def _options_to_openai(options: dict | None) -> dict:
    """Translate Ollama options dict -> OpenAI/llama-server params.

    ``num_ctx`` is intentionally dropped — context is fixed at server start.
    """
    out: dict[str, Any] = {}
    if not options:
        return out
    if "temperature" in options:
        out["temperature"] = options["temperature"]
    if "top_p" in options:
        out["top_p"] = options["top_p"]
    if "top_k" in options:
        out["top_k"] = options["top_k"]
    if "num_predict" in options:
        out["max_tokens"] = options["num_predict"]
    if "repeat_penalty" in options:
        out["repeat_penalty"] = options["repeat_penalty"]
    if "seed" in options:
        out["seed"] = options["seed"]
    return out


def _strip_assistant_extras(messages: list[dict]) -> list[dict]:
    """OpenAI-compat strict mode rejects ``thinking`` and unknown keys on
    assistant messages, and requires tool_calls items to have specific
    shape. Build a minimal copy."""
    cleaned: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            content = m.get("content", "")
            entry["content"] = content if content else ""
            tc = m.get("tool_calls") or []
            if tc:
                norm = []
                for i, t in enumerate(tc):
                    fn = getattr(t, "function", None) or t.get("function", {})
                    name = getattr(fn, "name", None) or (
                        fn.get("name") if isinstance(fn, dict) else None
                    )
                    args = getattr(fn, "arguments", None)
                    if args is None and isinstance(fn, dict):
                        args = fn.get("arguments")
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    elif isinstance(args, str):
                        args_str = args
                    else:
                        args_str = "{}"
                    tid = (
                        getattr(t, "id", None)
                        or (t.get("id") if isinstance(t, dict) else None)
                        or f"call_{i}"
                    )
                    norm.append(
                        {
                            "id": tid,
                            "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }
                    )
                entry["tool_calls"] = norm
            cleaned.append(entry)
        elif role == "tool":
            cleaned.append(
                {
                    "role": "tool",
                    "content": m.get("content", ""),
                    "tool_call_id": m.get("tool_call_id", "call_0"),
                }
            )
        else:
            cleaned.append({"role": role, "content": m.get("content", "")})
    return cleaned


class _ThinkSplitter:
    """Streaming splitter that routes content inside ``<think>...</think>``
    tags into the thinking channel and the rest into the content channel.

    Qwen3 reasoning models emit ``<think>``/``</think>`` inline via the chat
    template. Ollama parses these out when ``think=True``; llama-server does
    not, so we do it ourselves to preserve the renderer's UX.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Push a content chunk; return (thinking_out, content_out)."""
        text = self.buf + chunk
        self.buf = ""
        thinking_out: list[str] = []
        content_out: list[str] = []

        while text:
            if self.in_think:
                idx = text.find(self.CLOSE)
                if idx == -1:
                    # Could be a partial close tag at the end; hold tail.
                    keep = min(len(self.CLOSE) - 1, len(text))
                    if keep and self.CLOSE.startswith(text[-keep:]):
                        thinking_out.append(text[:-keep])
                        self.buf = text[-keep:]
                    else:
                        thinking_out.append(text)
                    text = ""
                else:
                    thinking_out.append(text[:idx])
                    text = text[idx + len(self.CLOSE) :]
                    self.in_think = False
            else:
                idx = text.find(self.OPEN)
                if idx == -1:
                    keep = min(len(self.OPEN) - 1, len(text))
                    if keep and self.OPEN.startswith(text[-keep:]):
                        content_out.append(text[:-keep])
                        self.buf = text[-keep:]
                    else:
                        content_out.append(text)
                    text = ""
                else:
                    content_out.append(text[:idx])
                    text = text[idx + len(self.OPEN) :]
                    self.in_think = True

        return "".join(thinking_out), "".join(content_out)


def _llama_chat_stream(
    *, model: str, messages: list[dict], tools, options
) -> Iterator[Any]:
    """Stream chunks shaped like Ollama's stream output."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": _strip_assistant_extras(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    payload.update(_options_to_openai(options))

    splitter = _ThinkSplitter()
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    prompt_tokens = 0
    eval_tokens = 0

    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
    with httpx.stream(
        "POST",
        f"{_LLAMA_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.lstrip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            usage = obj.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or 0
                eval_tokens = usage.get("completion_tokens", eval_tokens) or 0

            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            delta = ch.get("delta") or {}

            # llama-server with --jinja emits thinking in a separate
            # `reasoning_content` field (Qwen3 chat template support).
            # Older / template-less paths inline <think>…</think> in content,
            # so we still run the splitter as a fallback.
            reasoning_piece = delta.get("reasoning_content") or ""
            if reasoning_piece:
                yield _ns(
                    {
                        "message": {
                            "thinking": reasoning_piece,
                            "content": "",
                            "tool_calls": [],
                        },
                        "done": False,
                        "prompt_eval_count": 0,
                        "eval_count": 0,
                    }
                )

            content_piece = delta.get("content") or ""
            if content_piece:
                t, c = splitter.feed(content_piece)
                if t:
                    yield _ns(
                        {
                            "message": {
                                "thinking": t,
                                "content": "",
                                "tool_calls": [],
                            },
                            "done": False,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                        }
                    )
                if c:
                    yield _ns(
                        {
                            "message": {
                                "thinking": "",
                                "content": c,
                                "tool_calls": [],
                            },
                            "done": False,
                            "prompt_eval_count": 0,
                            "eval_count": 0,
                        }
                    )

            # Tool-call accumulation (OpenAI streams these in fragments).
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls_acc.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    # Emit final tool-call payload (parsed args) and stats. Build the
    # tool-call objects manually so that ``arguments`` stays a real ``dict``
    # (Ollama's SDK keeps it that way; downstream code does ``**arguments``).
    final_tcs = []
    for idx in sorted(tool_calls_acc):
        slot = tool_calls_acc[idx]
        try:
            args_dict = json.loads(slot["arguments"]) if slot["arguments"] else {}
        except json.JSONDecodeError:
            args_dict = {}
        final_tcs.append(
            SimpleNamespace(
                id=slot["id"] or f"call_{idx}",
                function=SimpleNamespace(name=slot["name"], arguments=args_dict),
            )
        )

    yield _ns(
        {
            "message": {
                "thinking": "",
                "content": "",
                "tool_calls": final_tcs,
            },
            "done": True,
            "prompt_eval_count": prompt_tokens,
            "eval_count": eval_tokens,
        }
    )


def _llama_chat_oneshot(*, model: str, messages: list[dict], tools, options) -> Any:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _strip_assistant_extras(messages),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    payload.update(_options_to_openai(options))

    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)
    r = httpx.post(f"{_LLAMA_URL}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    # llama-server with --jinja already separates reasoning from content.
    # Fallback: also run the splitter on `content` for inline <think> tags.
    splitter = _ThinkSplitter()
    inline_thinking, visible = splitter.feed(content)
    thinking = (reasoning or "") + (inline_thinking or "")

    raw_tcs = msg.get("tool_calls") or []
    tcs: list = []
    for i, t in enumerate(raw_tcs):
        fn = t.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        tcs.append(
            SimpleNamespace(
                id=t.get("id") or f"call_{i}",
                function=SimpleNamespace(name=fn.get("name"), arguments=args or {}),
            )
        )

    usage = obj.get("usage") or {}
    return _ns(
        {
            "message": {
                "thinking": thinking,
                "content": visible,
                "tool_calls": tcs,
            },
            "done": True,
            "prompt_eval_count": usage.get("prompt_tokens", 0) or 0,
            "eval_count": usage.get("completion_tokens", 0) or 0,
        }
    )


def _llama_show(model: str):
    """Mimic ``ollama.show`` enough that ``get_context_window`` can read it."""
    try:
        r = httpx.get(f"{_LLAMA_URL}/props", timeout=5.0)
        r.raise_for_status()
        props = r.json()
    except Exception:
        return _ns({"modelinfo": {}})

    # llama-server's /props exposes n_ctx as the runtime context size.
    n_ctx = (
        props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx")
        or 0
    )
    info: dict[str, Any] = {}
    if n_ctx:
        info["context_length"] = int(n_ctx)
    return _ns({"modelinfo": info})


# ---------------------------------------------------------------------------
# Public API (Ollama-compatible)
# ---------------------------------------------------------------------------


def chat(
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    think: bool = True,  # accepted for compat; thinking is split inline
    stream: bool = False,
    keep_alive: Any = None,  # accepted for compat; no-op for llama-server
    options: dict | None = None,
):
    if _BACKEND == "ollama":
        return _ollama_chat(
            model=model,
            messages=messages,
            tools=tools,
            think=think,
            stream=stream,
            keep_alive=keep_alive,
            options=options,
        )
    if stream:
        return _llama_chat_stream(
            model=model, messages=messages, tools=tools, options=options
        )
    return _llama_chat_oneshot(
        model=model, messages=messages, tools=tools, options=options
    )


def show(model: str):
    if _BACKEND == "ollama":
        return _ollama_show(model)
    return _llama_show(model)
