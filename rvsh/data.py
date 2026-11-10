"""Fixed, seeded GSM8K subsets."""
from __future__ import annotations

import random

from datasets import load_dataset

from .gsm8k import gold_answer


def test_subset(n: int, seed: int = 20261107) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [{"id": f"test-{i}", "question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])} for i in idx[:n]]


def train_pool(n: int, seed: int = 7) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [{"id": f"train-{i}", "question": ds[i]["question"], "solution": ds[i]["answer"],
             "gold": gold_answer(ds[i]["answer"])} for i in idx[:n]]
