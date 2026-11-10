"""Verifiable rewards for GRPO. Correctness is checked against the GSM8K gold number."""
from __future__ import annotations

from .gsm8k import has_answer_format, is_correct


def _text(c) -> str:
    if isinstance(c, list):  # conversational completions: [{"role": "assistant", "content": ...}]
        return "".join(m.get("content", "") for m in c)
    return c


def correctness_reward(completions, gold, **_) -> list[float]:
    return [1.0 if is_correct(_text(c), g) else 0.0 for c, g in zip(completions, gold)]


def format_reward(completions, **_) -> list[float]:
    return [0.1 if has_answer_format(_text(c)) else 0.0 for c in completions]
