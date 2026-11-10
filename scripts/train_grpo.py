"""(a) GRPO with verifiable rewards (TRL + LoRA). Writes results/train_grpo.json; adapter goes to checkpoints/ (gitignored)."""
import argparse, json, time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rvsh import data, prompts
from rvsh.compute import ComputeLedger
from rvsh.model import BASE_MODEL, load, non_embedding_params
from rvsh.rewards import correctness_reward, format_reward

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--prompts-per-step", type=int, default=2)
ap.add_argument("--generations", type=int, default=6)
ap.add_argument("--max-completion", type=int, default=256)
ap.add_argument("--out", default="results/train_grpo.json")
a = ap.parse_args()

pool = data.train_pool(a.steps * a.prompts_per_step)
ds = Dataset.from_list([{"prompt": prompts.cot_messages(x["question"]), "gold": x["gold"]} for x in pool])
model, tok = load(dtype=torch.bfloat16)
n_params = non_embedding_params(model)
cfg = GRPOConfig(
    output_dir="checkpoints/grpo", max_steps=a.steps, learning_rate=5e-5, lr_scheduler_type="constant",
    per_device_train_batch_size=a.prompts_per_step * a.generations, gradient_accumulation_steps=1,
    num_generations=a.generations, max_completion_length=a.max_completion, temperature=0.8,
    beta=0.0, logging_steps=1, save_strategy="no", report_to=[], seed=20261107, bf16=False,
    use_cpu=False, gradient_checkpointing=True, dataloader_pin_memory=False, shuffle_dataset=False,
)
trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[correctness_reward, format_reward],
                      args=cfg, train_dataset=ds, peft_config=LoraConfig(**LORA))
t0 = time.time()
trainer.train()
secs = time.time() - t0
trainer.model.save_pretrained("checkpoints/grpo/adapter")
hist = trainer.state.log_history
num_tokens = max(h.get("num_tokens", 0) for h in hist)
led = ComputeLedger(n_params=n_params, wall_seconds=secs)
# generation forward + old-logprob forward (2 x 2N) and LoRA fwd+bwd (4N) on every sampled token
led.add_inference(2 * num_tokens)
led.add_lora_train(num_tokens)
led.notes.append("GRPO FLOPs = num_tokens x (2N generate + 2N logprob + 4N LoRA fwd/bwd); attention FLOPs ignored")
curve = [{"step": h["step"], "reward": h.get("reward"), "correct": h.get("rewards/correctness_reward/mean"),
          "completion_len": h.get("completions/mean_length")} for h in hist if "reward" in h]
Path(a.out).write_text(json.dumps({"method": "grpo", "base_model": BASE_MODEL, "lora": LORA, "args": vars(a),
                                   "train_prompts": len(ds), "num_tokens": num_tokens, "compute": led.to_dict(),
                                   "curve": curve}, indent=2))
print(json.dumps(led.to_dict(), indent=2))
