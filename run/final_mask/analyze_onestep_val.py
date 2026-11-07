"""A Step 2 variant vs Ref on the va_val contexts: success, paired difference (95% CI by bootstrap over the 300
contexts, as run/variant_a/analyze_val_eval.py), per task family. Writes <dir>/summary.json.

    python run/final_mask/analyze_onestep_val.py --variant ioudpo --dir run/final_mask/ioudpo_val
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--variant", default="onestep")
p.add_argument("--dir", default=f"{RUN}/final_mask/onestep_val")
args = p.parse_args()


def load(pattern, variant):
    out, task = {}, {}
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line)
            if r["variant"] == variant:
                out[(r["episode"], r["perturb"])] = float(r["success"])
                task[r["episode"]] = r["task"]
    return out, task


def family(t):
    if t.startswith("push_") and "drawer" not in t:
        return "push"
    if any(x in t for x in ("lift", "place", "stack")):
        return "lift/place/stack"
    if "rotate" in t:
        return "rotate"
    return "articulated"


ref, task = load(f"{RUN}/variant_a/val/rollouts/per_sample_*.jsonl", "ref")
gt, _ = load(f"{RUN}/variant_a/val/rollouts/per_sample_*.jsonl", "gt")
new, _ = load(f"{args.dir}/rollouts/per_sample_*.jsonl", args.variant)
keys = sorted(set(ref) & set(new))
eps = sorted({e for e, _ in keys})
ctx = lambda s: np.array([np.mean([s[k] for k in keys if k[0] == e]) for e in eps])
a, b = ctx(new), ctx(ref)
boot = np.random.default_rng(0).integers(0, len(eps), size=(10000, len(eps)))
diff = (a - b)[boot].mean(1)
res = {"n_rollouts": len(keys), "n_contexts": len(eps), "ref": float(np.mean([ref[k] for k in keys])),
       args.variant: float(np.mean([new[k] for k in keys])), "diff": float((a - b).mean()),
       "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
       "wins": sum(new[k] > ref[k] for k in keys), "losses": sum(new[k] < ref[k] for k in keys),
       "gt_goal": float(np.mean(list(gt.values())))}
by = defaultdict(list)
for i, e in enumerate(eps):
    by[family(task[e])].append(i)
res["family"] = {f: {"n": len(i), "ref": float(b[i].mean()), args.variant: float(a[i].mean())} for f, i in sorted(by.items())}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{args.dir}/summary.json", "w"), indent=1)
