"""
Critic: self-review of the final drafted response.

The critic runs in-conversation (per user preference — no separate LLM
call). The orchestrator appends a ``[SYSTEM CRITIC]`` user message; the
model's very next turn is expected to emit a JSON object we can parse
into an ``{accept, issues}`` decision.

If ``accept=false``, the orchestrator appends a revision prompt and
loops. ``CRITIC_MAX_ROUNDS`` caps the loop.

This module is pure logic (no Ollama) so it's easy to unit test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent import prompts


@dataclass
class CriticVerdict:
    accept: bool
    issues: list[str]
    raw: str  # original response text (for debugging)
    parsed: bool  # False if we couldn't find valid JSON at all


_JSON_RE = re.compile(r"\{[^{}]*\"accept\"\s*:\s*(?:true|false)[^{}]*\}", re.DOTALL)


def parse_verdict(response_text: str) -> CriticVerdict:
    """
    Extract the critic's JSON verdict from its response.

    Strategy:
      1. Find the LAST JSON-ish block mentioning ``"accept"``.
      2. Try ``json.loads`` on it directly.
      3. If that fails, fall back to a truthiness heuristic — accept only if
         the response starts with CRITIC_ACCEPT or contains ``"accept": true``
         without ``false`` later.

    Never raises.
    """
    text = response_text or ""
    matches = list(_JSON_RE.finditer(text))
    for m in reversed(matches):
        blob = m.group(0)
        try:
            data = json.loads(blob)
            accept = bool(data.get("accept", False))
            issues = data.get("issues", []) or []
            if not isinstance(issues, list):
                issues = [str(issues)]
            return CriticVerdict(
                accept=accept,
                issues=[str(x) for x in issues],
                raw=text,
                parsed=True,
            )
        except json.JSONDecodeError:
            continue

    # No parseable JSON found.
    stripped = text.strip()
    if stripped.upper().startswith("CRITIC_ACCEPT"):
        return CriticVerdict(accept=True, issues=[], raw=text, parsed=False)
    return CriticVerdict(
        accept=False,
        issues=["critic did not return valid JSON"],
        raw=text,
        parsed=False,
    )


def build_injection(round_num: int, max_rounds: int) -> str:
    return prompts.critic(round_num=round_num, max_rounds=max_rounds)


def build_revision(issues: list[str]) -> str:
    return prompts.critic_revise(issues)
