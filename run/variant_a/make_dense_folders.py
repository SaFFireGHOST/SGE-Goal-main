"""A-dense conditions: CALVIN-EVALS-style folders for every Variant A training demo (splits.json `train`).

CALVIN-EVALS/va_dense/src/calvin_abcd_test_data/<ep+1>/ original_image.png (start frame, 200 -> 208 bilinear,
as make_pilot_contexts.py) + edit_instruction.txt. run/variant_a/run_dense_masks.sh adds the learned M_i
(va_dense/cond) and Step 2's M_f, which become the Step 3 conditions for L_beh (behavior_loss.BehaviorWindows).
The pilot's demos reuse their m0a0 M_i / M_f (same model, seed 1000) instead of being regenerated.
"""
import glob
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{RUN}/calvin/d_train"
DST = f"{RUN}/CALVIN-EVALS/va_dense/src/calvin_abcd_test_data"
SIZE = 208

with open(f"{D}/splits.json") as f:
    train = set(json.load(f)["train"])
n = 0
for part in sorted(glob.glob(f"{D}/contexts_part*.parquet")):
    for r in pq.read_table(part, columns=["episode_index", "instruction", "start_image"]).to_pylist():
        if r["episode_index"] not in train:
            continue
        d = os.path.join(DST, str(r["episode_index"] + 1))
        if os.path.isfile(os.path.join(d, "edit_instruction.txt")):
            n += 1
            continue
        os.makedirs(d, exist_ok=True)
        img = Image.open(io.BytesIO(r["start_image"])).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        img.save(os.path.join(d, "original_image.png"))
        with open(os.path.join(d, "edit_instruction.txt"), "w") as f:
            f.write(r["instruction"].strip() + "\n")
        n += 1
print(f"{n} demo folders in {DST}")
