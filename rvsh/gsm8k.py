"""GSM8K parsing and answer checking (the verifiable reward)."""
from __future__ import annotations

import re

_NUM = r"-?\d[\d,]*(?:\.\d+)?"


def normalize_number(s: str | None) -> str | None:
    """Turn '1,234.50' / '$12' / '7.0' into a canonical string ('1234.5', '12', '7')."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def gold_answer(solution: str) -> str:
    """Gold answer from a GSM8K solution ('... #### 72')."""
    tail = solution.split("####")[-1]
    out = normalize_number(tail)
    if out is None:
        raise ValueError(f"no gold number in: {tail!r}")
    return out


def strip_calc_annotations(solution: str) -> str:
    """Remove GSM8K '<<48/2=24>>' calculator annotations."""
    return re.sub(r"<<[^>]*>>", "", solution)


def solution_to_target(solution: str) -> str:
    """GSM8K solution -> SFT target in the same format the eval prompt asks for."""
    body, _, _ = solution.partition("####")
    body = strip_calc_annotations(body).strip()
    return f"{body}\nThe answer is {gold_answer(solution)}."


def extract_answer(text: str) -> str | None:
    """Model answer: 'The answer is N' > '\\boxed{N}' > '#### N' > last number in text."""
    for pat in (rf"answer is[:\s]*\$?\s*({_NUM})", r"\\boxed\{\s*\$?(" + _NUM + r")\s*\}", rf"####\s*\$?({_NUM})"):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return normalize_number(m[-1])
    nums = re.findall(_NUM, text)
    return normalize_number(nums[-1]) if nums else None


def is_correct(text: str, gold: str) -> bool:
    pred = extract_answer(text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def has_answer_format(text: str) -> bool:
    return re.search(rf"answer is[:\s]*\$?\s*{_NUM}", text, flags=re.IGNORECASE) is not None
