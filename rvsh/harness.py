"""Condition (c): no training, better harness.
1) few-shot CoT, 2) program-of-thought run through the safe calculator tool,
3) self-check: if they disagree, draw 2 sampled CoT votes and take the majority."""
from __future__ import annotations

from collections import Counter

from . import calc, gsm8k, prompts


def pot_answer(code: str) -> str | None:
    try:
        return gsm8k.normalize_number(calc.format_number(calc.run_program(code)))
    except (calc.CalcError, ZeroDivisionError, OverflowError, ValueError, TypeError, RecursionError):
        return None


def vote(cot: str | None, pot: str | None, extra: list[str | None]) -> tuple[str | None, str]:
    """Combine answers. Returns (answer, route)."""
    if cot is not None and cot == pot:
        return cot, "agree"
    votes = Counter(a for a in [cot, pot, *extra] if a is not None)
    if not votes:
        return None, "none"
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1] and pot is not None and votes[pot] == top[0][1]:
        return pot, "tie->pot"
    return top[0][0], "majority"


def run_harness(model, tok, questions: list[str], max_new_tokens: int = 320, batch_size: int = 16):
    """Returns (answers, routes, raw dict, tokens, seconds)."""
    from .model import generate
    cot_txt, t1, s1 = generate(model, tok, [prompts.fewshot_cot_messages(q) for q in questions], max_new_tokens, batch_size)
    pot_txt, t2, s2 = generate(model, tok, [prompts.pot_messages(q) for q in questions], 256, batch_size)
    cot = [gsm8k.extract_answer(t) for t in cot_txt]
    pot = [pot_answer(t) for t in pot_txt]
    disagree = [i for i in range(len(questions)) if not (cot[i] is not None and cot[i] == pot[i])]
    extra: dict[int, list] = {i: [] for i in disagree}
    t3 = s3 = 0
    for k in range(2):
        if not disagree:
            break
        txt, tt, ss = generate(model, tok, [prompts.fewshot_cot_messages(questions[i]) for i in disagree],
                               max_new_tokens, batch_size, temperature=0.7, seed=1000 * (k + 1))
        t3 += tt
        s3 += ss
        for i, t in zip(disagree, txt):
            extra[i].append(gsm8k.extract_answer(t))
    answers, routes = [], []
    for i in range(len(questions)):
        a, r = vote(cot[i], pot[i], extra.get(i, []))
        answers.append(a)
        routes.append(r)
    raw = {"cot": cot_txt, "pot": pot_txt, "cot_ans": cot, "pot_ans": pot, "extra": extra}
    return answers, routes, raw, t1 + t2 + t3, s1 + s2 + s3
