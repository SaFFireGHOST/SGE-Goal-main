#!/usr/bin/env python3
"""
upload_weights.py — Upload model weights and eval-data to Modal volumes.

This populates the shared 'lgie-eval-weights' and 'lgie-eval-data' volumes
that all evaluation runs reference.

Usage:
    modal run upload_weights.py
"""
import modal
import os
import time

app = modal.App("lgie-upload-weights")

weights_volume = modal.Volume.from_name("lgie-eval-weights", create_if_missing=True)
eval_data_volume = modal.Volume.from_name("lgie-eval-data", create_if_missing=True)

BASE = os.path.dirname(os.path.abspath(__file__))


@app.local_entrypoint()
def main():
    t0 = time.time()

    # --- 1. Upload model weights ---
    print("=" * 60)
    print("  Uploading model weights to 'lgie-eval-weights' volume")
    print("=" * 60)

    # Check if weights already present
    try:
        existing = {e.path.strip("/") for e in weights_volume.listdir("/")}
        has_sam = "sam_vit_h_4b8939.pth" in existing
    except Exception:
        existing = set()
        has_sam = False

    if has_sam:
        print("  Weights already on volume — skipping (delete volume to re-upload)")
    else:
        with weights_volume.batch_upload(force=True) as batch:
            # SAM weights
            sam_path = os.path.join(BASE, "sam_vit_h_4b8939.pth")
            if os.path.exists(sam_path):
                print(f"  Uploading SAM weights ({os.path.getsize(sam_path) / 1e9:.1f} GB) ...")
                batch.put_file(sam_path, "/sam_vit_h_4b8939.pth")

            # XMem weights
            xmem_dir = os.path.join(BASE, "XMem", "saves")
            if os.path.isdir(xmem_dir):
                for f in os.listdir(xmem_dir):
                    if f.endswith(".pth"):
                        src = os.path.join(xmem_dir, f)
                        print(f"  Uploading XMem/{f} ({os.path.getsize(src) / 1e6:.0f} MB) ...")
                        batch.put_file(src, f"/XMem/{f}")

            # LoFTR weights
            loftr_dir = os.path.join(BASE, "loftr_weights")
            if os.path.isdir(loftr_dir):
                for f in os.listdir(loftr_dir):
                    if f.endswith(".ckpt"):
                        src = os.path.join(loftr_dir, f)
                        print(f"  Uploading loftr_weights/{f} ({os.path.getsize(src) / 1e6:.0f} MB) ...")
                        batch.put_file(src, f"/loftr_weights/{f}")

        print("  ✓ Weights uploaded!")

    # --- 2. Upload lgie-eval-data ---
    print()
    print("=" * 60)
    print("  Uploading eval-data to 'lgie-eval-data' volume")
    print("=" * 60)

    # Check if eval-data already present
    try:
        existing_eval = {e.path.strip("/") for e in eval_data_volume.listdir("/")}
        has_eval = "lgie-eval-data" in existing_eval or "original_image" in existing_eval
    except Exception:
        has_eval = False

    if has_eval:
        print("  Eval-data already on volume — skipping")
    else:
        eval_data_path = os.path.join(BASE, "lgie-eval-data")
        if os.path.isdir(eval_data_path):
            print(f"  Uploading lgie-eval-data/ ...")
            with eval_data_volume.batch_upload(force=True) as batch:
                batch.put_directory(eval_data_path, "/lgie-eval-data")
            print("  ✓ Eval-data uploaded!")
        else:
            print(f"  ERROR: {eval_data_path} not found!")

    elapsed = time.time() - t0
    print(f"\nAll uploads complete in {elapsed:.0f}s.")
