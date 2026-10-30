"""Upload the learned initial-mask (Step 1) EVF-SAM checkpoints to the private HF model repo SaFFire/sge-goal.

One commit: calvin-initial-mask-evf-{union,arm-object}/ (best.pt, args.json, metrics.jsonl, test_metrics.json)
plus a README section. Then checks each remote file's SHA-256 against the local file.
Token: the cached HF login (huggingface-cli login). Usage: python run/upload_ier_to_hf.py
"""
import hashlib
import os

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "SaFFire/sge-goal"
RUN = os.path.dirname(os.path.abspath(__file__))
MODELS = {"evf_union": "calvin-initial-mask-evf-union", "evf_arm_object": "calvin-initial-mask-evf-arm-object"}
FILES = ("best.pt", "args.json", "metrics.jsonl", "test_metrics.json")
MARKER = "## Learned Initial Mask (Step 1, EVF-SAM)"
SECTION = f"""

---

{MARKER}

Replaces the training-free Step 1 (initial edit region mask `M_i` from `(I_o, T)`). EVF-SAM with SAM ViT-H
frozen; BEiT-3-L, projection and mask decoder fine-tuned (BCE + Dice, AdamW lr 2e-5 BEiT / 1e-4 head,
3 epochs, batch 8) on the 22k HF CALVIN train tuples, best checkpoint by validation IoU (epoch 2, step 8250).

| Folder | Prompt | Test IoU (1087 rows) | IoU > 0.5 |
|---|---|---|---|
| `calvin-initial-mask-evf-union/` | instruction only | 0.715 | 84.7% |
| `calvin-initial-mask-evf-arm-object/` | arm prompt + instruction, soft union | 0.711 | 84.1% |
| training-free Step 1 (paper) | — | 0.102 | — |

Downstream (frozen GR-MG, all 1087 CALVIN test rows, released Step 2 + Step 3), CL1..CL5 success %:

| `M_i` | CL1 | CL2 | CL3 | CL4 | CL5 |
|---|---|---|---|---|---|
| learned `union` | 62.74 | 36.40 | 14.38 | 5.16 | 2.55 |
| training-free | 53.17 | 26.09 | 9.34 | 3.47 | 1.84 |

Each folder: `best.pt` (fp16/bf16 trainable weights), `args.json` (training config), `metrics.jsonl`
(training log), `test_metrics.json`. Inference: `run/predict_ier.sh --checkpoint <folder>/best.pt
--data-dir <CALVIN-EVALS folder> --out-dir <out>` (`sge_goal/grounding/learned_ier/predict_ier_mask.py`).
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


api = HfApi()
readme = open(hf_hub_download(REPO, "README.md")).read()
ops = [CommitOperationAdd(f"{dst}/{f}", f"{RUN}/ier_training/{src}/{f}") for src, dst in MODELS.items() for f in FILES]
if MARKER not in readme:
    ops.append(CommitOperationAdd("README.md", (readme.rstrip() + SECTION).encode()))
print(f"uploading {len(ops)} files to {REPO}", flush=True)
commit = api.create_commit(REPO, ops, commit_message="Add learned initial-mask (Step 1) EVF-SAM checkpoints")
print("commit", commit.oid, flush=True)

info = api.repo_info(REPO, files_metadata=True)
remote = {s.rfilename: s for s in info.siblings}
ok = True
for src, dst in MODELS.items():
    for f in FILES:
        local, r = f"{RUN}/ier_training/{src}/{f}", remote.get(f"{dst}/{f}")
        if r is None:
            print("MISSING", f"{dst}/{f}"); ok = False; continue
        match = (r.lfs.sha256 == sha256(local)) if r.lfs else (r.size == os.path.getsize(local))
        print(f"{'OK ' if match else 'BAD'} {dst}/{f} ({r.size} bytes)")
        ok &= match
print("UPLOAD_VERIFIED" if ok else "UPLOAD_MISMATCH")
