"""Variant A held-out evaluation setup: all 300 `val` contexts of run/calvin/d_train/splits.json.

These episodes are never used for DPO pairs (pilot = `train`) or A-dense windows (`train`), so they compare
Ref / A-sparse / A-dense checkpoints without selection bias. Same folder layout as make_pilot_contexts.py:
  CALVIN-EVALS/va_val/src/calvin_abcd_test_data/<ep+1>/  original_image.png (200 -> 208 bilinear) + edit_instruction.txt
  CALVIN-EVALS/va_val/gt/calvin_abcd_test_data/<ep+1>/   same + edited_image.png = the demo's end frame
Writes run/variant_a/val/contexts.json (sorted episode indices). run/variant_a/run_val_eval.sh does the rest.
"""
import glob
import io
import json
import os

import pyarrow.parquet as pq
from PIL import Image

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{RUN}/calvin/d_train"
DST = f"{RUN}/CALVIN-EVALS/va_val"
OUT = f"{RUN}/variant_a/val"
SIZE = 208

with open(f"{D}/splits.json") as f:
    val = set(json.load(f)["val"])
t = pq.read_table(sorted(glob.glob(f"{D}/contexts_part*.parquet")),
                  columns=["episode_index", "task", "instruction", "start_image", "end_image"])
ctx = {r["episode_index"]: r for r in t.to_pylist() if r["episode_index"] in val}
assert len(ctx) == len(val), (len(ctx), len(val))

for e, r in sorted(ctx.items()):
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
    json.dump(sorted(ctx), f)
print(f"{len(ctx)} contexts over {len({r['task'] for r in ctx.values()})} tasks -> {DST}/{{src,gt}}")
