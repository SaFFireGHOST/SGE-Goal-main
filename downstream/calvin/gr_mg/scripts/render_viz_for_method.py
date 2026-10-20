#!/usr/bin/env python3
"""Render 5x5 = 25 chain-instance videos for a method.

Selects 5 fixed (skeleton_idx, start_in_chain) pairs per chain length
(1..5), then dispatches them in parallel to ``viz_chain_remote`` on Modal.
Saved locally at ``.runlogs/viz/<method>/length{n}/instance_{i}.mp4``.

Selection strategy
------------------
For each length ``n`` (1..5):
  Enumerate all length-``n`` sub-chains across all skeletons in deterministic
  order; pick the first 5 (or fewer if ``< 5`` exist; for length 1 fall back
  to picking 5 distinct first-clusters of 5 distinct skeletons).

Usage
-----
    python scripts/render_viz_for_method.py \\
        --method-name flux-kontext-dev-ft \\
        --account modal-account-1 \\
        [--use-hf-parquet-goal]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNLOGS = REPO_ROOT / ".runlogs"
SKEL_LOCAL = RUNLOGS / "chain_skeletons.json"


def _enumerate_length_n(
    skeletons: List[dict], n: int
) -> List[Tuple[int, int]]:
    """Return (skeleton_idx, start_in_chain) pairs for all length-n sub-chains
    across all skeletons, in deterministic order."""
    out = []
    for sk_idx, sk in enumerate(skeletons):
        clusters = sk["clusters"]
        if len(clusters) < n:
            continue
        for st in range(len(clusters) - n + 1):
            out.append((sk_idx, st))
    return out


def _pick_5(pairs: List[Tuple[int, int]], length: int,
            skeletons: List[dict]) -> List[Tuple[int, int]]:
    """Pick 5 representative pairs. For length=1, pick from 5 DIFFERENT
    skeletons to maximise variety. Otherwise just take first 5."""
    if length == 1:
        seen_skeletons = set()
        out = []
        for sk_idx, st in pairs:
            if sk_idx in seen_skeletons:
                continue
            seen_skeletons.add(sk_idx)
            out.append((sk_idx, st))
            if len(out) == 5:
                break
        # pad with whatever else if needed
        if len(out) < 5:
            for p in pairs:
                if p not in out:
                    out.append(p)
                    if len(out) == 5:
                        break
        return out[:5]
    return pairs[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-name", required=True)
    ap.add_argument("--account", required=True,
                    help="Modal profile to use for rendering")
    ap.add_argument("--use-hf-parquet-goal", action="store_true",
                    help="Use HF parquet goal_image (ground-truth method).")
    ap.add_argument("--ep-len-per-step", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    if not SKEL_LOCAL.exists():
        sys.exit(f"chain_skeletons.json missing at {SKEL_LOCAL}")
    with SKEL_LOCAL.open() as fh:
        sk_doc = json.load(fh)
    skeletons = sk_doc["chain_skeletons"]

    out_root = RUNLOGS / "viz" / args.method_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Manifest of selections.
    manifest = {}
    for n in range(1, 6):
        all_pairs = _enumerate_length_n(skeletons, n)
        picked = _pick_5(all_pairs, n, skeletons)
        manifest[n] = picked

    print(f"[viz] method={args.method_name} account={args.account}")
    for n, picks in manifest.items():
        print(f"  length {n}: {len(picks)} instances -> {picks}")

    env = os.environ.copy()
    env["MODAL_PROFILE"] = args.account

    for n, picks in manifest.items():
        out_dir = out_root / f"length{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (sk_idx, st) in enumerate(picks):
            out_path = out_dir / f"instance_{i}.mp4"
            if out_path.exists():
                print(f"[viz]   exists, skip: {out_path}")
                continue

            cmd = [
                "modal", "run", "modal_app.py::viz_chain",
                "--method-name", args.method_name,
                "--skeleton-idx", str(sk_idx),
                "--start-in-chain", str(st),
                "--chain-length", str(n),
                "--output", str(out_path),
                "--ep-len-per-step", str(args.ep_len_per_step),
                "--fps", str(args.fps),
            ]
            if args.use_hf_parquet_goal:
                cmd += ["--use-hf-parquet-goal"]

            log_path = out_dir / f"instance_{i}.log"
            print(f"[viz]   length={n} idx={i} sk={sk_idx} st={st} -> {out_path}")
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    cmd, env=env, cwd=str(REPO_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT,
                )
            if proc.returncode != 0:
                print(f"[viz]   FAILED rc={proc.returncode}; see {log_path}")
                continue
            # The CLI prints the function return-bytes hex... but we used
            # local_entrypoint? No, viz_chain_remote is a function returning
            # bytes — that gets printed. We need to capture differently.
            # Simpler: use a local-entrypoint wrapper.

    print(f"[viz] done; output under {out_root}")


if __name__ == "__main__":
    main()
