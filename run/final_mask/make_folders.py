"""Final-mask (Step 2) diagnostics: CALVIN-EVALS-style folders.

1. Oracle  CALVIN-EVALS/sge_goal_union_gtmf/calvin_abcd_test_data/<N>/  (N = test row + 1)
   = sge_goal_union_ier (learned M_i, same original / instruction) with the ground-truth final mask from the
   HF test parquet as final_edit_region_mask.png (raw parquet bytes, the format Step 3 was trained on).
   The predicted final_edit_region_mask.jpg is not copied: Step 3's find_image prefers .jpg over .png.
2. Step 2 sampling  CALVIN-EVALS/fer_diag/<set>/calvin_abcd_test_data/<k>/  (k = 1..256)
   val_pred_s0..s7  256 random HF validation tuples, learned M_i (pred_ier/validation.parquet), one folder
                    per Step 2 seed (best-of-N / medoid / vote)
   val_gt_s0        same tuples, ground-truth M_i                      } overfitting vs ambiguity:
   train_gt_s0      256 random HF train tuples, ground-truth M_i       } Step 2 IoU on train vs val
   run/final_mask/fer_diag_index.json maps each set's k to (split, row, id) for analyze_fer_diag.py.
"""
import glob
import io
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = f"{RUN}/calvin/hf_dataset"
SUB = "calvin_abcd_test_data"
N_SAMPLE, N_SEEDS, SEED = 256, 8, 0

# 1. oracle folder
src = f"{RUN}/CALVIN-EVALS/sge_goal_union_ier/{SUB}"
dst = f"{RUN}/CALVIN-EVALS/sge_goal_union_gtmf/{SUB}"
gt_f = pq.read_table(glob.glob(f"{HF}/data/test-*.parquet"), columns=["final_edit_region_mask"])["final_edit_region_mask"]
gt_f = gt_f.to_pylist()
for i, cell in enumerate(gt_f):
    s, d = f"{src}/{i + 1}", f"{dst}/{i + 1}"
    os.makedirs(d, exist_ok=True)
    for name in ("original_image.png", "edit_instruction.txt", "initial_edit_region_mask.jpg"):
        shutil.copy2(f"{s}/{name}", f"{d}/{name}")
    with open(f"{d}/final_edit_region_mask.png", "wb") as f:
        f.write(cell["bytes"])
    assert not os.path.exists(f"{d}/final_edit_region_mask.jpg")
print(f"oracle: {len(gt_f)} cases -> {dst}")


# 2. Step 2 sampling folders
def load(split):
    return pq.read_table(sorted(glob.glob(f"{HF}/data/{split}-*.parquet")),
                         columns=["id", "original_image", "initial_edit_region_mask", "final_edit_region_mask",
                                  "edit_instruction"]).to_pylist()


val, train = load("validation"), load("train")
pred = {r["id"]: r["pred_initial_mask"] for r in pq.read_table(f"{HF}/pred_ier/validation.parquet").to_pylist()}
rng = np.random.default_rng(SEED)
rows = {"validation": sorted(rng.choice(len(val), N_SAMPLE, replace=False).tolist()),
        "train": sorted(rng.choice(len(train), N_SAMPLE, replace=False).tolist())}


def write(set_name, table, split, use_pred):
    base = f"{RUN}/CALVIN-EVALS/fer_diag/{set_name}/{SUB}"
    for k, row in enumerate(rows[split], start=1):
        r, d = table[row], f"{base}/{k}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/original_image.png", "wb") as f:
            f.write(r["original_image"]["bytes"])
        with open(f"{d}/edit_instruction.txt", "w") as f:
            f.write(r["edit_instruction"].strip() + "\n")
        m = pred[r["id"]] if use_pred else r["initial_edit_region_mask"]["bytes"]
        with open(f"{d}/initial_edit_region_mask.{'jpg' if use_pred else 'png'}", "wb") as f:
            f.write(m["bytes"] if isinstance(m, dict) else m)
    return {k: {"split": split, "row": row, "id": table[row]["id"]} for k, row in enumerate(rows[split], start=1)}


index = {}
for s in range(N_SEEDS):
    index[f"val_pred_s{s}"] = write(f"val_pred_s{s}", val, "validation", True)
index["val_gt_s0"] = write("val_gt_s0", val, "validation", False)
index["train_gt_s0"] = write("train_gt_s0", train, "train", False)
os.makedirs(f"{RUN}/final_mask", exist_ok=True)
with open(f"{RUN}/final_mask/fer_diag_index.json", "w") as f:
    json.dump(index, f)
print(f"Step 2 sampling: {len(index)} sets x {N_SAMPLE} cases -> {RUN}/CALVIN-EVALS/fer_diag")
