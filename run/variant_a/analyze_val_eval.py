"""Summarise run/variant_a/run_val_eval.sh: success per variant and paired differences vs ref.

Paired over contexts (same start states / masks / seeds); 95% CI by bootstrap over the 300 contexts.
Writes run/variant_a/val/summary.json.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = f"{RUN}/variant_a/val"
rows = [json.loads(l) for f in sorted(glob.glob(f"{P}/rollouts/per_sample_*.jsonl")) for l in open(f)]
succ = defaultdict(dict)  # variant -> {(ep, perturb): success}
task = {}
for r in rows:
    succ[r["variant"]][(r["episode"], r["perturb"])] = float(r["success"])
    task[r["episode"]] = r["task"]
eps = sorted(task)
ctx = {v: np.array([np.mean([s[(e, k)] for k in range(2) if (e, k) in s]) for e in eps]) for v, s in succ.items()}

rng = np.random.default_rng(0)
boot = rng.integers(0, len(eps), size=(10000, len(eps)))
out = {"n_contexts": len(eps), "variants": {}}
for v in ["ref", "asparse", "adense", "gt"]:
    vals = list(succ[v].values())
    d = {"n": len(vals), "success": float(np.mean(vals))}
    if v != "ref":
        diff = ctx[v] - ctx["ref"]
        bs = diff[boot].mean(1)
        both = [(succ[v][key], succ["ref"][key]) for key in succ[v] if key in succ["ref"]]
        d.update(diff_vs_ref=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 p_le_0=float((bs <= 0).mean()),
                 wins=sum(a > b for a, b in both), losses=sum(a < b for a, b in both))
    out["variants"][v] = d
    print(v, json.dumps(d))

by_task = defaultdict(list)
for i, e in enumerate(eps):
    by_task[task[e]].append(i)
print("\nper task (n contexts): ref / asparse / adense / gt")
pt = {}
for t, idx in sorted(by_task.items(), key=lambda x: -len(x[1])):
    pt[t] = {v: float(ctx[v][idx].mean()) for v in ["ref", "asparse", "adense", "gt"]}
    print(f"{t:40s} {len(idx):3d}  " + "  ".join(f"{pt[t][v]:.2f}" for v in ["ref", "asparse", "adense", "gt"]))
out["per_task"] = pt
json.dump(out, open(f"{P}/summary.json", "w"), indent=1)
