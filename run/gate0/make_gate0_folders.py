"""Gate 0 setup: pick 300 test contexts (stratified by task) and lay out 4 goal-sample folders.

  CALVIN-EVALS/gate0/v0/calvin_abcd_test_data/<row+1>/  full copy of the learned-IER pipeline output (seed 0)
  CALVIN-EVALS/gate0/v{1,2,3}/...                        original image + instruction + learned M_i only;
                                                         Steps 2-3 are re-sampled there with other seeds.
The learned M_i is deterministic, so goal samples differ only through the Step 2 / Step 3 seeds.
Writes run/gate0/rows.json (sorted 0-indexed HF rows).
"""
import json
import os
import random
import shutil
from collections import defaultdict

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{RUN}/CALVIN-EVALS/sge_goal_union_ier/calvin_abcd_test_data"
DST = f"{RUN}/CALVIN-EVALS/gate0"
N_CONTEXTS, N_VARIANTS, SEED = 300, 4, 0

with open(f"{RUN}/eval_inputs/hf_calvin_mapping.json") as f:
    task_of = {m["hf_row"]: m["task"] for m in json.load(f)["matches"]}
by_task = defaultdict(list)
for row, task in sorted(task_of.items()):
    by_task[task].append(row)

# Proportional allocation per task (largest remainder), at least one row per task.
total = len(task_of)
quota = {t: max(1, int(N_CONTEXTS * len(r) / total)) for t, r in by_task.items()}
rema = sorted(by_task, key=lambda t: N_CONTEXTS * len(by_task[t]) / total - quota[t], reverse=True)
for t in rema[: N_CONTEXTS - sum(quota.values())]:
    quota[t] += 1
rng = random.Random(SEED)
rows = sorted(r for t, rs in by_task.items() for r in rng.sample(rs, min(quota[t], len(rs))))
assert len(rows) == N_CONTEXTS, len(rows)

for k in range(N_VARIANTS):
    for row in rows:
        src = os.path.join(SRC, str(row + 1))
        dst = os.path.join(DST, f"v{k}", "calvin_abcd_test_data", str(row + 1))
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(src):
            keep = k == 0 or name.startswith("original_image.") or name in (
                "edit_instruction.txt", "initial_edit_region_mask.jpg")
            if keep:
                shutil.copy2(os.path.join(src, name), dst)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows.json"), "w") as f:
    json.dump(rows, f)
print(f"{len(rows)} contexts over {len(by_task)} tasks; folders v0..v{N_VARIANTS - 1} in {DST}")
