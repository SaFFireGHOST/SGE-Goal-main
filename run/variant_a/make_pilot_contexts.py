"""A-sparse pilot setup: pick 300 Variant A training contexts and lay out their CALVIN-EVALS folders.

Contexts come from the `train` split of run/calvin/d_train/splits.json (CALVIN D training, never
test), stratified by task like the Gate 0 selection. Folder <episode_index + 1> holds:
  CALVIN-EVALS/va_pilot/src/calvin_abcd_test_data/<ep+1>/  original_image.png (start frame, 200 -> 208
                                                           bilinear, the HF image size) + edit_instruction.txt
  CALVIN-EVALS/va_pilot/gt/calvin_abcd_test_data/<ep+1>/   same + edited_image.png = the demo's end frame
                                                           (ground-truth goal, sanity reference for rollouts)
The learned M_i and the Step 2 / Step 3 samples are added by run/variant_a/run_pilot.sh.
Writes run/variant_a/pilot/contexts.json (sorted episode indices).
"""
import glob
import io
import json
import os
import random
from collections import defaultdict

import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{RUN}/calvin/d_train"
DST = f"{RUN}/CALVIN-EVALS/va_pilot"
OUT = f"{RUN}/variant_a/pilot"
N_CONTEXTS, SEED, SIZE = 300, 0, 208

with open(f"{D}/splits.json") as f:
    train = set(json.load(f)["train"])
t = pq.read_table(sorted(glob.glob(f"{D}/contexts_part*.parquet")),
                  columns=["episode_index", "task", "instruction", "start_image", "end_image"])
ctx = {r["episode_index"]: r for r in t.to_pylist() if r["episode_index"] in train}
by_task = defaultdict(list)
for e, r in sorted(ctx.items()):
    by_task[r["task"]].append(e)

# Proportional allocation per task (largest remainder), at least one context per task.
total = len(ctx)
quota = {k: max(1, int(N_CONTEXTS * len(v) / total)) for k, v in by_task.items()}
order = sorted(by_task, key=lambda k: N_CONTEXTS * len(by_task[k]) / total - quota[k], reverse=True)
for k in order[: N_CONTEXTS - sum(quota.values())]:
    quota[k] += 1
rng = random.Random(SEED)
eps = sorted(e for k, v in by_task.items() for e in rng.sample(v, quota[k]))
assert len(eps) == N_CONTEXTS, len(eps)

for e in eps:
    r = ctx[e]
    start = Image.open(io.BytesIO(r["start_image"])).convert("RGB")
    for variant in ("src", "gt"):
        d = os.path.join(DST, variant, "calvin_abcd_test_data", str(e + 1))
        os.makedirs(d, exist_ok=True)
        start.resize((SIZE, SIZE), Image.BILINEAR).save(os.path.join(d, "original_image.png"))
        with open(os.path.join(d, "edit_instruction.txt"), "w") as f:
            f.write(r["instruction"].strip() + "\n")
        if variant == "gt":
            Image.open(io.BytesIO(r["end_image"])).convert("RGB").save(os.path.join(d, "edited_image.png"))

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/contexts.json", "w") as f:
    json.dump(eps, f)
print(f"{len(eps)} contexts over {len(by_task)} tasks -> {DST}/{{src,gt}}")
