"""Gate 0 analysis: mixed-outcome rate, cross-fitted headroom of the best goal, rollout noise.

Proposal §6 gate: cross-fitted gain of the best goal >= 5 points AND >= 30% of contexts mixed.
Cross-fitting: pick each context's best goal on start state A, score it on start state B
(and B -> A), so the winner's curse of noisy selection does not inflate the gain.
Ties in the selection half are scored by the average of the tied goals (= random tie-break).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
OUT = f"{RUN}/eval_gate0"

succ = defaultdict(dict)  # (row) -> {(variant, perturb): 0/1}
task_of = {}
for f in glob.glob(f"{OUT}/per_sample_*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        succ[r["hf_row"]][(r["variant"], r["perturb"])] = int(r["success"])
        task_of[r["hf_row"]] = r["task"]
rows = sorted(succ)
variants = sorted({v for d in succ.values() for v, _ in d})
perturbs = sorted({k for d in succ.values() for _, k in d})
assert len(perturbs) == 2, "cross-fitting below assumes 2 start states"
complete = [r for r in rows if len(succ[r]) == len(variants) * len(perturbs)]
print(f"contexts: {len(complete)} complete / {len(rows)} seen; variants {variants}; start states {perturbs}")
S = np.array([[[succ[r][(v, k)] for k in perturbs] for v in variants] for r in complete])  # [C, G, 2]
C, G, _ = S.shape


def cross_fit_gain(S):
    gains, chosen = [], []
    for a, b in ((0, 1), (1, 0)):
        sel = S[:, :, a]
        best = sel == sel.max(axis=1, keepdims=True)  # tied winners
        chosen_b = (S[:, :, b] * best).sum(1) / best.sum(1)
        chosen.append(chosen_b)
        gains.append(chosen_b - S[:, :, b].mean(1))
    return np.mean(gains, axis=0), np.mean(chosen, axis=0)


per_ctx_gain, per_ctx_chosen = cross_fit_gain(S)
rng = np.random.default_rng(0)
boot = [per_ctx_gain[rng.integers(0, C, C)].mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

goal_rate = S.mean(2)                                  # per-goal success over start states
mixed_any = ((S.sum((1, 2)) > 0) & (S.sum((1, 2)) < G * 2)).mean()
mixed_goals = (goal_rate.max(1) > goal_rate.min(1)).mean()
flip = (S[:, :, 0] != S[:, :, 1]).mean()

result = {
    "contexts": int(C), "goals_per_context": int(G), "start_states": 2,
    "single_goal_SR": float(S.mean()),
    "SR_by_start_state": [float(S[:, :, k].mean()) for k in range(2)],
    "best_of_goals_naive": float(S.max(1).mean()),
    "best_goal_cross_fitted_SR": float(per_ctx_chosen.mean()),
    "headroom_cross_fitted_pts": float(100 * per_ctx_gain.mean()),
    "headroom_95ci_pts": [float(100 * lo), float(100 * hi)],
    "mixed_outcomes_any_rollout": float(mixed_any),
    "mixed_outcomes_between_goals": float(mixed_goals),
    "rollout_flip_rate_same_goal": float(flip),
}

# Determinism check: v0 at the original reset re-runs the learned-IER eval exactly.
prev = {}
for f in glob.glob(f"{RUN}/eval_sge_goal_union_ier/per_sample_*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        prev[r["hf_row"]] = int(r["success"])
if "v0" in variants and prev:
    same = [succ[r][("v0", 0)] == prev[r] for r in complete if r in prev]
    result["determinism_v0_p0_vs_previous_eval"] = float(np.mean(same))

result["gate0_pass"] = bool(result["headroom_cross_fitted_pts"] >= 5 and result["mixed_outcomes_between_goals"] >= 0.30)

print(json.dumps(result, indent=2))
print("\nper task: contexts  single-goal SR  mixed-between-goals  cross-fitted gain (pts)")
for t in sorted(set(task_of[r] for r in complete)):
    idx = [i for i, r in enumerate(complete) if task_of[r] == t]
    print(f"  {t:28s} {len(idx):3d}   {S[idx].mean():.2f}   "
          f"{(goal_rate[idx].max(1) > goal_rate[idx].min(1)).mean():.2f}   {100 * per_ctx_gain[idx].mean():+.1f}")
with open(f"{OUT}/gate0_result.json", "w") as f:
    json.dump(result, f, indent=2)
