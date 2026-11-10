"""Evaluate one condition on the fixed GSM8K test subset. Writes results/eval_<name>.json."""
import argparse, json
from pathlib import Path

from rvsh import data, gsm8k, prompts
from rvsh.compute import ComputeLedger
from rvsh.harness import run_harness
from rvsh.model import BASE_MODEL, generate, load, non_embedding_params

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, choices=["base", "sft", "grpo", "harness"])
ap.add_argument("--adapter", default=None)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--max-new", type=int, default=320)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--out-dir", default="results")
a = ap.parse_args()

items = data.test_subset(a.n)
model, tok = load(adapter=a.adapter)
led = ComputeLedger(n_params=non_embedding_params(model))
qs = [x["question"] for x in items]
if a.name == "harness":
    answers, routes, raw, tokens, secs = run_harness(model, tok, qs, a.max_new, a.batch)
    texts = raw["cot"]
else:
    texts, tokens, secs = generate(model, tok, [prompts.cot_messages(q) for q in qs], a.max_new, a.batch)
    answers, routes = [gsm8k.extract_answer(t) for t in texts], ["single"] * len(qs)
led.add_inference(tokens)
led.wall_seconds = secs
per = []
for x, ans, r, t in zip(items, answers, routes, texts):
    ok = ans is not None and abs(float(ans) - float(x["gold"])) < 1e-6
    per.append({"id": x["id"], "gold": x["gold"], "pred": ans, "correct": ok, "route": r,
                "formatted": gsm8k.has_answer_format(t), "text_tail": t[-200:]})
correct = sum(p["correct"] for p in per)
summary = {"condition": a.name, "base_model": BASE_MODEL, "adapter": a.adapter, "n": len(per), "correct": correct,
           "accuracy": correct / len(per), "format_rate": sum(p["formatted"] for p in per) / len(per),
           "inference": led.to_dict(), "flops_per_question": led.flops / len(per)}
if a.name == "harness":
    from collections import Counter
    summary["routes"] = dict(Counter(routes))
Path(a.out_dir, f"eval_{a.name}.json").write_text(json.dumps({"summary": summary, "items": per}, indent=2))
print(json.dumps(summary, indent=2))
