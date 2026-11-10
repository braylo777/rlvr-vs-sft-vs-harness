"""(b) SFT on GSM8K reference solutions (LoRA), stopped when its FLOPs match the GRPO run."""
import argparse, json, math, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from rvsh import data, prompts
from rvsh.compute import ComputeLedger, sft_tokens_for_budget
from rvsh.gsm8k import solution_to_target
from rvsh.model import BASE_MODEL, load, non_embedding_params

ap = argparse.ArgumentParser()
ap.add_argument("--match", default="results/train_grpo.json", help="GRPO result whose FLOPs we match")
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", default="results/train_sft.json")
a = ap.parse_args()

budget = json.loads(Path(a.match).read_text())["compute"]["flops"]
model, tok = load(dtype=torch.bfloat16)
n_params = non_embedding_params(model)
token_budget = sft_tokens_for_budget(budget, n_params)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(**LORA))
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
dev = next(model.parameters()).device

# Disjoint from the GRPO prompts (GRPO used the first items of the same shuffled pool).
pool = data.train_pool(4000)[2000:]
tok.padding_side = "right"


def encode(x):
    msgs = prompts.cot_messages(x["question"])
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = p + solution_to_target(x["solution"]) + tok.eos_token
    p_ids, f_ids = tok(p)["input_ids"], tok(full)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels


used, step, losses, t0, i = 0, 0, [], time.time(), 0
while used < token_budget and i < len(pool):
    batch = [encode(x) for x in pool[i:i + a.batch]]
    i += a.batch
    L = max(len(b[0]) for b in batch)
    ids = torch.tensor([b[0] + [tok.pad_token_id] * (L - len(b[0])) for b in batch], device=dev)
    lab = torch.tensor([b[1] + [-100] * (L - len(b[1])) for b in batch], device=dev)
    att = torch.tensor([[1] * len(b[0]) + [0] * (L - len(b[0])) for b in batch], device=dev)
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward()
    opt.step(); opt.zero_grad()
    used += int(att.sum()); step += 1
    losses.append(round(loss.item(), 4))
    if step % 10 == 0 or step <= 3:
        print(f"step {step} tokens {used}/{token_budget} loss {sum(losses[-10:])/10:.3f}", flush=True)
secs = time.time() - t0
model.save_pretrained("checkpoints/sft/adapter")
led = ComputeLedger(n_params=n_params, wall_seconds=secs)
led.add_lora_train(used)
led.notes.append("SFT FLOPs = tokens x 4N (LoRA fwd/bwd); attention FLOPs ignored")
Path(a.out).write_text(json.dumps({"method": "sft", "base_model": BASE_MODEL, "lora": LORA, "args": vars(a),
                                   "examples": i, "steps": step, "token_budget": token_budget, "tokens_used": used,
                                   "matched_flops": budget, "compute": led.to_dict(), "loss_curve": losses}, indent=2))
print(json.dumps(led.to_dict(), indent=2))
