"""A-sparse pilot: rollouts -> stage-wise advantages (proposal Eq. 8) -> Flow-DPO preference pairs.

Per context c (folders m<k>a<j> = mask k, appearance j; 2 start states each):
  p(x)   = success rate of goal x over its start states
  U(M_k) = mean p over the mask's appearances;  b(c) = mean U over masks
  A2 = U(M_k) - b(c)  (Step 2),  A3 = p(x) - U(M_k)  (Step 3)
Pairs: Step 2 masks with U(M_w) - U(M_l) >= --delta2, conditioned on (I_o, T, M_i);
       Step 3 appearances under the same M_f with p(x_w) - p(x_l) >= --delta3, conditioned on (I_o, T, M_i, M_f).
10% of contexts (seeded) are held out: their pairs measure held-out DPO preference accuracy.

Writes run/variant_a/pilot/pairs.json ({"step2"|"step3": {"train"|"val": [pair, ...]}}) and pilot_stats.json.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = f"{RUN}/variant_a/pilot"
V = f"{RUN}/CALVIN-EVALS/va_pilot"
SUB = "calvin_abcd_test_data"


def find(d, stem):
    for ext in ("jpg", "png", "jpeg"):
        if os.path.isfile(f"{d}/{stem}.{ext}"):
            return f"{d}/{stem}.{ext}"
    raise FileNotFoundError(f"{d}/{stem}.*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta2", type=float, default=0.5)
    ap.add_argument("--delta3", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    p = defaultdict(lambda: defaultdict(list))  # p[ep][variant] -> successes
    task = {}
    for f in glob.glob(f"{P}/rollouts/per_sample_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            p[r["episode"]][r["variant"]].append(float(r["success"]))
            task[r["episode"]] = r["task"]
    eps = sorted(e for e in p if all(len(p[e][f"m{k}a{j}"]) == 2 for k in (0, 1) for j in (0, 1)))
    val_eps = set(random.Random(0).sample(eps, round(args.val_frac * len(eps))))

    pairs = {s: {"train": [], "val": []} for s in ("step2", "step3")}
    A2, A3, mixed = [], [], 0
    for e in eps:
        rate = {v: float(np.mean(p[e][v])) for v in p[e]}
        U = [np.mean([rate[f"m{k}a0"], rate[f"m{k}a1"]]) for k in (0, 1)]
        b = float(np.mean(U))
        mixed += int(0 < np.mean([rate[f"m{k}a{j}"] for k in (0, 1) for j in (0, 1)]) < 1)
        split = "val" if e in val_eps else "train"
        d = {k: {j: f"{V}/m{k}a{j}/{SUB}/{e + 1}" for j in (0, 1)} for k in (0, 1)}
        instr = open(f"{d[0][0]}/edit_instruction.txt").read().strip()
        base = {"episode": e, "task": task[e], "instruction": instr,
                "original": find(d[0][0], "original_image"), "m_i": find(d[0][0], "initial_edit_region_mask")}
        for k in (0, 1):
            A2.append(U[k] - b)
            for j in (0, 1):
                A3.append(rate[f"m{k}a{j}"] - U[k])
        if abs(U[0] - U[1]) >= args.delta2:
            w, l = (0, 1) if U[0] > U[1] else (1, 0)
            pairs["step2"][split].append({**base, "win": find(d[w][0], "final_edit_region_mask"),
                                          "lose": find(d[l][0], "final_edit_region_mask"),
                                          "score_w": U[w], "score_l": U[l]})
        for k in (0, 1):
            r0, r1 = rate[f"m{k}a0"], rate[f"m{k}a1"]
            if abs(r0 - r1) >= args.delta3:
                w, l = (0, 1) if r0 > r1 else (1, 0)
                pairs["step3"][split].append({**base, "m_f": find(d[k][0], "final_edit_region_mask"),
                                              "win": find(d[k][w], "edited_image"),
                                              "lose": find(d[k][l], "edited_image"),
                                              "score_w": rate[f"m{k}a{w}"], "score_l": rate[f"m{k}a{l}"]})

    sr = {v: float(np.mean([s for e in eps for s in p[e].get(v, [])])) for v in ("m0a0", "m0a1", "m1a0", "m1a1", "gt")}
    stats = {"contexts": len(eps), "val_contexts": len(val_eps), "success_rate": sr,
             "mixed_context_frac": mixed / max(len(eps), 1),
             "mean_abs_A2": float(np.mean(np.abs(A2))), "mean_abs_A3": float(np.mean(np.abs(A3))),
             "pairs": {s: {k: len(v) for k, v in pairs[s].items()} for s in pairs},
             "delta2": args.delta2, "delta3": args.delta3}
    with open(f"{P}/pairs.json", "w") as f:
        json.dump(pairs, f, indent=1)
    with open(f"{P}/pilot_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
