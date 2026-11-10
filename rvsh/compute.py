"""Compute accounting. All FLOP counts are the standard dense-transformer estimates:
forward ~= 2*N per token, full fine-tune fwd+bwd ~= 6*N per token, LoRA fwd+bwd ~= 4*N per token
(frozen base weights skip weight-gradient matmuls). N = non-embedding parameter count.
Dollar costs are ASSUMPTIONS stated in ASSUMED_GPU and are labeled as such in every report."""
from __future__ import annotations

from dataclasses import dataclass, field

ASSUMED_GPU = {"name": "A100-80GB (assumed)", "effective_tflops": 100.0, "usd_per_hour": 2.0}


@dataclass
class ComputeLedger:
    n_params: int
    fwd_tokens: int = 0
    lora_train_tokens: int = 0
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def add_inference(self, tokens: int) -> None:
        self.fwd_tokens += int(tokens)

    def add_lora_train(self, tokens: int) -> None:
        self.lora_train_tokens += int(tokens)

    @property
    def flops(self) -> float:
        return 2.0 * self.n_params * self.fwd_tokens + 4.0 * self.n_params * self.lora_train_tokens

    def to_dict(self) -> dict:
        return {"n_params": self.n_params, "fwd_tokens": self.fwd_tokens,
                "lora_train_tokens": self.lora_train_tokens, "flops": self.flops,
                "pflops": self.flops / 1e15, "wall_seconds_this_mac": round(self.wall_seconds, 1),
                "usd_assumed_gpu": usd_cost(self.flops), "notes": self.notes}


def usd_cost(flops: float, gpu: dict = ASSUMED_GPU) -> float:
    seconds = flops / (gpu["effective_tflops"] * 1e12)
    return seconds / 3600.0 * gpu["usd_per_hour"]


def sft_tokens_for_budget(target_flops: float, n_params: int) -> int:
    """How many LoRA training tokens SFT may use to match a FLOP budget."""
    return int(target_flops / (4.0 * n_params))


def break_even_queries(train_flops: float, extra_flops_per_query: float) -> float:
    """Queries after which a per-query harness overhead exceeds a one-time training cost."""
    if extra_flops_per_query <= 0:
        return float("inf")
    return train_flops / extra_flops_per_query
