#!/usr/bin/env python3
"""Post-eval pipeline: detect finished experiments, pull results, render viz.

For every (account, method) configured in ACCOUNT_PLAN, this script
periodically checks whether the corresponding experiment's per-shard sentinel
files exist on the runs volume. When all 5 (or 10 for ground_truth) shards
are done, it:

  1. Pulls ``/runs/eval_method/<method>/per_sample_*.jsonl`` and
     ``shard_done_*.json`` to ``.runlogs/results/<method>/``.
  2. Records the method as 'pulled'.
  3. Renders 5 chain instances per chain length (1..5) = 25 videos using
     ``modal_app.py::viz_chain``, saved at
     ``.runlogs/viz/<method>/length<n>/instance_<i>.mp4``.

Once all 11 methods are pulled, runs ``aggregate_chain_success.py`` to
produce the final 11x5 table at ``.runlogs/final_table.csv``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNLOGS = REPO_ROOT / ".runlogs"
ORCH_DIR = RUNLOGS / "orchestrator"
RESULTS_DIR = RUNLOGS / "results"
VIZ_DIR = RUNLOGS / "viz"
SKEL_PATH = RUNLOGS / "chain_skeletons.json"

# Mirror of orchestrator.ACCOUNT_PLAN, plus modal-account-1 mapping.
ACCOUNT_METHODS: Dict[str, List[str]] = {
    "modal-account-1":       ["flux-kontext-dev"],
    "modal-account-2":      ["flux-kontext-dev-ft", "flux2-klein"],
    "modal-account-3":   ["flux2-klein-ft", "gemini-3-pro-image"],
    "modal-account-4": ["gpt-image-1.5", "clover"],
    "modal-account-5":     ["qwen-image-edit", "susie"],
    "modal-account-6":       ["ground_truth"],
}

ALL_METHODS_ORDERED = [
    "ground_truth",
    "clover",
    "flux-kontext-dev",
    "flux-kontext-dev-ft",
    "flux2-klein",
    "flux2-klein-ft",
    "gemini-3-pro-image",
    "gpt-image-1.5",
    "qwen-image-edit",
    "susie",
]
NUM_SHARDS_GT = 10
NUM_SHARDS_DEFAULT = 5


def _modal_env(account: str) -> Dict[str, str]:
    e = os.environ.copy()
    e["MODAL_PROFILE"] = account
    return e


def _modal_run(account: str, args: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["modal"] + args
    with log_path.open("a") as logf:
        logf.write(f"\n=== {datetime.utcnow().isoformat()}Z $ {' '.join(cmd)}\n")
        logf.flush()
        return subprocess.run(
            cmd, env=_modal_env(account), cwd=str(REPO_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode


def _shard_ranges(num_shards: int) -> List[Tuple[int, int]]:
    return [
        (1087 * i // num_shards, 1087 * (i + 1) // num_shards)
        for i in range(num_shards)
    ]


def _is_eval_done(account: str, method: str) -> bool:
    """Check if all expected shard sentinels exist on the runs volume."""
    num_shards = NUM_SHARDS_GT if method == "ground_truth" else NUM_SHARDS_DEFAULT
    sentinel_local = ORCH_DIR / f"{account}__{method}__eval_done.OK"
    if sentinel_local.exists():
        return True
    log_path = ORCH_DIR / f"{account}__{method}__check.log"
    expected_sentinels = [
        f"shard_done_{s}_{e}.json" for s, e in _shard_ranges(num_shards)
    ]
    # Use modal CLI to list runs/eval_method/<method>/.
    cmd = ["modal", "volume", "ls", "rm-policy-runs",
           f"eval_method/{method}/"]
    try:
        out = subprocess.run(
            cmd, env=_modal_env(account), cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:
        print(f"[post] WARN volume ls failed for {account}/{method}: {e}")
        return False
    if out.returncode != 0:
        return False
    listing = out.stdout
    if not all(s in listing for s in expected_sentinels):
        return False
    sentinel_local.touch()
    return True


def _pull_method_results(account: str, method: str) -> bool:
    """Pull per_sample_*.jsonl + shard_done_*.json into .runlogs/results/<method>/."""
    sentinel_local = ORCH_DIR / f"{account}__{method}__pulled.OK"
    if sentinel_local.exists():
        return True
    out_dir = RESULTS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ORCH_DIR / f"{account}__{method}__pull.log"
    rc = _modal_run(
        account,
        ["volume", "get", "--force", "rm-policy-runs",
         f"eval_method/{method}/", str(out_dir)],
        log_path,
    )
    if rc != 0:
        print(f"[post] WARN pull failed for {account}/{method}")
        return False
    # Modal puts files under <out_dir>/<method>/... — flatten if needed.
    nested = out_dir / method
    if nested.exists():
        for p in nested.iterdir():
            p.rename(out_dir / p.name)
        nested.rmdir()
    # Validate: count total per_sample rows across all shards.
    total_rows = 0
    for sp in sorted(out_dir.glob("per_sample_*.jsonl")):
        with sp.open() as fh:
            total_rows += sum(1 for _ in fh)
    # Also validate shard_done files have evaluated > 0 (not all-skipped stale data).
    num_shards = NUM_SHARDS_GT if method == "ground_truth" else NUM_SHARDS_DEFAULT
    for s, e in _shard_ranges(num_shards):
        sd_file = out_dir / f"shard_done_{s}_{e}.json"
        if sd_file.exists():
            with sd_file.open() as fh:
                sd = json.load(fh)
            if sd.get("evaluated", 0) == 0 and sd.get("skipped", 0) > 0:
                print(f"[post] WARN stale shard_done for {method} [{s},{e}): all skipped, "
                      f"cleaning and retrying later")
                # Clean up stale pulled data.
                import shutil
                shutil.rmtree(out_dir)
                return False
    if total_rows < 1087:
        print(f"[post] WARN {method}: only {total_rows}/1087 rows pulled; "
              f"cleaning and retrying later")
        import shutil
        shutil.rmtree(out_dir)
        return False
    sentinel_local.touch()
    print(f"[post] pulled {account}/{method} -> {out_dir} ({total_rows} rows)")
    return True


def _enumerate_length_n(skeletons: List[dict], n: int) -> List[Tuple[int, int]]:
    out = []
    for sk_idx, sk in enumerate(skeletons):
        clusters = sk["clusters"]
        if len(clusters) < n:
            continue
        for st in range(len(clusters) - n + 1):
            out.append((sk_idx, st))
    return out


def _pick_5(pairs: List[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
    if length == 1:
        seen = set()
        out = []
        for sk_idx, st in pairs:
            if sk_idx in seen:
                continue
            seen.add(sk_idx)
            out.append((sk_idx, st))
            if len(out) == 5:
                break
        if len(out) < 5:
            for p in pairs:
                if p not in out:
                    out.append(p)
                    if len(out) == 5:
                        break
        return out[:5]
    return pairs[:5]


def _render_viz_for(account: str, method: str) -> None:
    sentinel_local = ORCH_DIR / f"{account}__{method}__viz_done.OK"
    if sentinel_local.exists():
        return
    print(f"[post] rendering viz: {account}/{method}")
    if not SKEL_PATH.exists():
        print(f"[post] WARN: skeletons file missing locally ({SKEL_PATH})")
        return
    with SKEL_PATH.open() as fh:
        skeletons = json.load(fh)["chain_skeletons"]

    use_hf_parquet = method == "ground_truth"
    out_root = VIZ_DIR / method
    out_root.mkdir(parents=True, exist_ok=True)

    for n in range(1, 6):
        all_pairs = _enumerate_length_n(skeletons, n)
        picks = _pick_5(all_pairs, n)
        out_dir = out_root / f"length{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (sk_idx, st) in enumerate(picks):
            out_path = out_dir / f"instance_{i}.mp4"
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            log_path = out_dir / f"instance_{i}.log"
            cmd = [
                "modal", "run", "modal_app.py::viz_chain",
                "--method-name", method,
                "--skeleton-idx", str(sk_idx),
                "--start-in-chain", str(st),
                "--chain-length", str(n),
                "--output", str(out_path),
                "--ep-len-per-step", "360",
                "--fps", "30",
            ]
            if use_hf_parquet:
                cmd += ["--use-hf-parquet-goal"]
            with log_path.open("w") as logf:
                logf.write(f"\n=== {datetime.utcnow().isoformat()}Z $ {' '.join(cmd)}\n")
                logf.flush()
                rc = subprocess.run(
                    cmd, env=_modal_env(account), cwd=str(REPO_ROOT),
                    stdout=logf, stderr=subprocess.STDOUT,
                ).returncode
            if rc != 0:
                print(f"[post] viz FAIL  {method}/length{n}/instance_{i} rc={rc}")
            else:
                print(f"[post]   wrote  {out_path}")
    sentinel_local.touch()


def _aggregate_final_table() -> None:
    cmd = [
        sys.executable, "-m", "evaluate.aggregate_chain_success",
        "--chain_skeletons", str(SKEL_PATH),
        "--results_dir", str(RESULTS_DIR),
        "--methods", *ALL_METHODS_ORDERED,
        "--out_csv", str(RUNLOGS / "final_table.csv"),
        "--out_json", str(RUNLOGS / "final_table.json"),
    ]
    print(f"[post] running aggregator: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    if rc == 0:
        print(f"[post] FINAL TABLE -> {RUNLOGS / 'final_table.csv'}")
    else:
        print(f"[post] aggregator failed rc={rc}")


def _run_viz_parallel() -> None:
    """Render viz for all methods in parallel (one subprocess per account)."""
    import concurrent.futures

    tasks = []
    for account, methods in ACCOUNT_METHODS.items():
        for method in methods:
            viz_sentinel = ORCH_DIR / f"{account}__{method}__viz_done.OK"
            if viz_sentinel.exists():
                continue
            tasks.append((account, method))

    if not tasks:
        print("[post] all viz already done")
        return

    print(f"[post] rendering viz for {len(tasks)} methods in parallel")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_render_viz_for, a, m): (a, m) for a, m in tasks}
        for fut in concurrent.futures.as_completed(futs):
            a, m = futs[fut]
            try:
                fut.result()
                print(f"[post] viz done: {a}/{m}")
            except Exception as e:
                print(f"[post] viz FAILED: {a}/{m}: {e}")


def main():
    print(f"[post] starting; methods: {ALL_METHODS_ORDERED}")

    # Phase 1: poll until all experiments finish and results are pulled
    while True:
        any_progress = False
        all_pulled = True
        for account, methods in ACCOUNT_METHODS.items():
            for method in methods:
                pulled_sentinel = ORCH_DIR / f"{account}__{method}__pulled.OK"
                if pulled_sentinel.exists():
                    continue
                if _is_eval_done(account, method):
                    if _pull_method_results(account, method):
                        any_progress = True
                    else:
                        all_pulled = False
                else:
                    all_pulled = False
        if all_pulled:
            break
        if not any_progress:
            time.sleep(60)

    # Phase 2: aggregate final table (does not need viz)
    print("[post] all methods pulled; running final aggregation")
    _aggregate_final_table()

    # Phase 3: render viz in parallel across accounts
    _run_viz_parallel()

    print("[post] ALL DONE")


if __name__ == "__main__":
    main()
