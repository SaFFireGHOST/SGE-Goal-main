"""IoU-ranked preference pairs for Flow-DPO on Step 2, from sample_step2.py's output (run_sample_step2.sh).

Per train tuple: winner = the Step 2 sample with the highest IoU against the ground-truth final mask, loser = the
lowest; kept only if the gap is >= --min-gap. Same conditions for both (original image, learned M_i,
instruction). Held-out pairs (for pair loss / accuracy during training) are whole episodes, ~--val-frac of rows.
Output in the trainer's format ({"step2": {"train": [...], "val": [...]}}, sge_goal/variant_a/flux_lora_trainer.py
--mode dpo --pairs):

    python run/final_mask/build_iou_pairs.py      # -> run/final_mask/iou_pairs.json
"""
import argparse
import json
import os
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def episode(i):
    return i.rsplit("_atomic_", 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=f"{HERE}/step2_samples")
    p.add_argument("--out", default=f"{HERE}/iou_pairs.json")
    p.add_argument("--min-gap", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.05)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(f"{args.samples}/samples.jsonl")]
    pairs = {"train": [], "val": []}
    for r in rows:
        iou = np.array(r["iou"])
        w, l = int(iou.argmax()), int(iou.argmin())
        if iou[w] - iou[l] < args.min_gap:
            continue
        d = f"{args.samples}/rows/{r['row']}"
        m_i = f"{d}/m_i.jpg" if os.path.exists(f"{d}/m_i.jpg") else f"{d}/m_i.png"
        # Episode-level split, stable under adding rows (hash of the episode id).
        split = "val" if zlib.crc32(episode(r["id"]).encode()) % 1000 < args.val_frac * 1000 else "train"
        pairs[split].append({"row": r["row"], "id": r["id"], "instruction": r["instruction"],
                             "original": f"{d}/original.png", "m_i": m_i,
                             "win": f"{d}/s{w}.png", "lose": f"{d}/s{l}.png",
                             "score_w": float(iou[w]), "score_l": float(iou[l])})
    json.dump({"step2": pairs, "min_gap": args.min_gap, "rows": len(rows)}, open(args.out, "w"), indent=1)

    for split, ps in pairs.items():
        w, l = np.array([q["score_w"] for q in ps]), np.array([q["score_l"] for q in ps])
        print(f"{split}: {len(ps)} pairs from {len({episode(q['id']) for q in ps})} episodes; "
              f"winner IoU {w.mean():.3f}, loser IoU {l.mean():.3f}, gap {np.mean(w - l):.3f}")
    print(f"kept {sum(map(len, pairs.values()))}/{len(rows)} rows (gap >= {args.min_gap}) -> {args.out}")


if __name__ == "__main__":
    main()
