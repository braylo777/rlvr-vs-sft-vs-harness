"""Model loading and batched chat generation with exact token accounting."""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_id: str = BASE_MODEL, adapter: str | None = None, dtype=torch.bfloat16, device: str | None = None):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    return model, tok


def non_embedding_params(model) -> int:
    emb = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    return total - emb


@torch.no_grad()
def generate(model, tok, conversations: list[list[dict]], max_new_tokens: int = 320, batch_size: int = 16,
             temperature: float = 0.0, seed: int = 0) -> tuple[list[str], int, float]:
    """Returns (completions, tokens_processed, seconds). tokens = non-pad prompt + generated tokens."""
    outs: list[str] = []
    tokens = 0
    t0 = time.time()
    device = next(model.parameters()).device
    for i in range(0, len(conversations), batch_size):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if temperature > 0:
            torch.manual_seed(seed + i)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        tokens += int(enc["attention_mask"].sum())
        for row in new:
            ids = row.tolist()
            eos = [j for j, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)]
            ids = ids[: eos[0] + 1] if eos else ids
            tokens += len(ids)
            outs.append(tok.decode(ids, skip_special_tokens=True))
    return outs, tokens, time.time() - t0
