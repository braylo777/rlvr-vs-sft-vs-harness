"""Build results/summary.json and results/summary.md from the committed result files only."""
import json, math
from pathlib import Path

from rvsh.compute import ASSUMED_GPU, break_even_queries, usd_cost

R = Path("results")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


grpo, sft = load("train_grpo.json"), load("train_sft.json")
ev = {k: load(f"eval_{k}.json") for k in ["base", "sft", "grpo", "harness"]}
base = ev["base"]["summary"]
rows = []
for k, e in ev.items():
    if not e:
        continue
    s = e["summary"]
    train = {"sft": sft, "grpo": grpo}.get(k)
    train_flops = train["compute"]["flops"] if train else 0.0
    extra_q = s["flops_per_question"] - base["flops_per_question"]
    lo, hi = wilson(s["correct"], s["n"])
    rows.append({"condition": k, "n": s["n"], "correct": s["correct"], "accuracy": s["accuracy"], "ci95": [lo, hi],
                 "delta_vs_base_pp": 100 * (s["accuracy"] - base["accuracy"]), "format_rate": s["format_rate"],
                 "train_flops": train_flops, "train_wall_s_mac": train["compute"]["wall_seconds_this_mac"] if train else 0,
                 "train_usd_assumed": usd_cost(train_flops), "infer_flops_per_q": s["flops_per_question"],
                 "extra_infer_flops_per_q": extra_q, "eval_wall_s_mac": s["inference"]["wall_seconds_this_mac"],
                 "routes": s.get("routes")})
hr = next((r for r in rows if r["condition"] == "harness"), None)
if hr and grpo:
    hr["break_even_queries_vs_grpo_training"] = break_even_queries(grpo["compute"]["flops"], hr["extra_infer_flops_per_q"])

# paired comparison vs base: discordant pairs (McNemar exact, two-sided)
def mcnemar(a, b):
    ia = {x["id"]: x["correct"] for x in a["items"]}
    b01 = sum(1 for x in b["items"] if not ia[x["id"]] and x["correct"])
    b10 = sum(1 for x in b["items"] if ia[x["id"]] and not x["correct"])
    n, k = b01 + b10, min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"gained": b01, "lost": b10, "p_value": p}

for r in rows:
    if r["condition"] != "base":
        r["vs_base"] = mcnemar(ev["base"], ev[r["condition"]])

out = {"assumed_gpu_for_usd": ASSUMED_GPU, "grpo_train": grpo and {k: grpo[k] for k in ("args", "train_prompts", "num_tokens")},
       "sft_train": sft and {k: sft[k] for k in ("examples", "steps", "tokens_used", "matched_flops")}, "rows": rows}
(R / "summary.json").write_text(json.dumps(out, indent=2))

md = ["| condition | accuracy (95% CI) | Δ vs base | gained/lost vs base (McNemar p) | train PFLOPs | train wall (this Mac) | train $ (assumed GPU) | inference GFLOPs / question |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    vb = r.get("vs_base")
    md.append(f"| {r['condition']} | {100*r['accuracy']:.1f}% ({100*r['ci95'][0]:.1f}–{100*r['ci95'][1]:.1f}%, n={r['n']}) | "
              f"{r['delta_vs_base_pp']:+.1f} pp | {'—' if not vb else f'{vb['gained']}/{vb['lost']} (p={vb['p_value']:.3f})'} | "
              f"{r['train_flops']/1e15:.3f} | {r['train_wall_s_mac']/60:.1f} min | ${r['train_usd_assumed']:.4f} | "
              f"{r['infer_flops_per_q']/1e9:.0f} |")
if hr and "break_even_queries_vs_grpo_training" in hr:
    md.append(f"\nHarness extra inference per question: {hr['extra_infer_flops_per_q']/1e9:.0f} GFLOPs. "
              f"Break-even vs the GRPO training cost: {hr['break_even_queries_vs_grpo_training']:.0f} questions. Routes: {hr['routes']}.")
(R / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
