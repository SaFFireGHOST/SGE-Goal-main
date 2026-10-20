#!/usr/bin/env python3
"""Multi-account Modal orchestrator for the 11x5 success-rate table.

Plan
----
* Account assignments (5 GPUs each method, 2 methods per account, in parallel):
    modal-account-1:        clover, flux-kontext-dev
    modal-account-2:       flux-kontext-dev-ft, flux2-klein
    modal-account-3:    flux2-klein-ft, gemini-3-pro-image
    modal-account-4:  gpt-image-1.5, clover
    modal-account-5:      qwen-image-edit, susie
    modal-account-6 (10 shards): ground_truth (HF parquet)

This script:
  * Bootstrap-uploads CALVIN-D, ckpts, mapping, parquet, eval-method dirs to
    each account on demand.
  * Launches the GPU evaluations in parallel (each as a background
    ``modal run --profile <p> modal_app.py::run_method ...`` process).
  * Tracks every running app in ``.runlogs/orchestrator/inflight.json`` and
    streams logs to ``.runlogs/orchestrator/<account>__<method>.log``.

Usage
-----
    python orchestrator.py bootstrap --account modal-account-1 \\
        --methods clover flux-kontext-dev
    python orchestrator.py launch --account modal-account-1 \\
        --methods clover flux-kontext-dev
    python orchestrator.py launch_all
    python orchestrator.py status
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]   # GR-MG/
WORKSPACE_ROOT = REPO_ROOT.parent                  # RM-Policy/
EVALS_ROOT = WORKSPACE_ROOT / "CALVIN-EVALS"
RUNLOGS = REPO_ROOT / ".runlogs"
ORCH_DIR = RUNLOGS / "orchestrator"
ORCH_DIR.mkdir(parents=True, exist_ok=True)
INFLIGHT = ORCH_DIR / "inflight.json"


# ---- Account / method assignment ------------------------------------------

ACCOUNT_PLAN: Dict[str, List[str]] = {
    "modal-account-1":       ["flux-kontext-dev"],
    "modal-account-2":      ["flux-kontext-dev-ft", "flux2-klein"],
    "modal-account-3":   ["flux2-klein-ft", "gemini-3-pro-image"],
    "modal-account-4": ["gpt-image-1.5", "clover"],
    "modal-account-5":     ["qwen-image-edit", "susie"],
    "modal-account-6":       ["ground_truth"],
}

# Default ckpt files we expect to already be uploaded once per account.
CKPT_FILES = [
    "gr_mg/epoch=47-step=83712.ckpt",
    "mae_pretrain_vit_base.pth",
]

NUM_SHARDS_DEFAULT = 5
NUM_SHARDS_GT = 10  # ground-truth account has 10 GPU quota


# ---- Inflight tracking -----------------------------------------------------

def _load_inflight() -> List[dict]:
    if not INFLIGHT.exists():
        return []
    with INFLIGHT.open() as fh:
        return json.load(fh)


def _save_inflight(items: List[dict]) -> None:
    with INFLIGHT.open("w") as fh:
        json.dump(items, fh, indent=2)


def _add_inflight(rec: dict) -> None:
    items = _load_inflight()
    items.append(rec)
    _save_inflight(items)


def _drop_inflight(pid: int) -> None:
    items = _load_inflight()
    items = [it for it in items if it.get("pid") != pid]
    _save_inflight(items)


# ---- Modal command runner --------------------------------------------------

def _modal_env(profile: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["MODAL_PROFILE"] = profile
    return env


def _run_modal_blocking(profile: str, args: List[str], log_path: Path) -> int:
    """Run ``modal <args>`` synchronously, tee to log_path."""
    cmd = ["modal"] + args
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[orch] [{profile}] $ {' '.join(shlex.quote(a) for a in cmd)}",
          flush=True)
    print(f"[orch] [{profile}]   logging -> {log_path}", flush=True)
    with log_path.open("a") as logf:
        logf.write(f"\n\n=== {datetime.utcnow().isoformat()}Z $ "
                   f"{' '.join(shlex.quote(a) for a in cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd, env=_modal_env(profile), cwd=str(REPO_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
        )
    return proc.returncode


def _run_modal_background(profile: str, args: List[str], tag: str) -> dict:
    """Spawn ``modal <args>`` as a detached process; return tracking record."""
    cmd = ["modal"] + args
    log_path = ORCH_DIR / f"{profile}__{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    print(f"[orch] [{profile}] launching background: {' '.join(shlex.quote(a) for a in cmd)}",
          flush=True)
    with log_path.open("a") as logf:
        logf.write(f"\n\n=== {datetime.utcnow().isoformat()}Z BG $ "
                   f"{' '.join(shlex.quote(a) for a in cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, env=_modal_env(profile), cwd=str(REPO_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    rec = {
        "pid": proc.pid,
        "profile": profile,
        "tag": tag,
        "cmd": cmd,
        "log": str(log_path),
        "started_at": datetime.utcnow().isoformat() + "Z",
    }
    _add_inflight(rec)
    return rec


# ---- Bootstrap -------------------------------------------------------------

def cmd_bootstrap(args):
    """Per account: ensure CALVIN data, ckpts, mapping, parquet, eval-data are present."""
    account = args.account
    methods = args.methods or ACCOUNT_PLAN[account]
    log_path = ORCH_DIR / f"{account}__bootstrap.log"

    # 1. CALVIN validation
    if not args.skip_calvin:
        rc = _run_modal_blocking(
            account, ["run", "modal_app.py::fetch_calvin_d"], log_path,
        )
        if rc != 0:
            sys.exit(f"[orch] {account}: fetch_calvin_d failed rc={rc}")

    # 2. Checkpoints — download directly from public URLs (avoid local upload).
    if not args.skip_ckpts:
        rc = _run_modal_blocking(
            account,
            ["run", "modal_app.py::fetch_grmg_ckpts",
             "--policy", "--no-goal-gen", "--no-mae"],
            log_path,
        )
        if rc != 0:
            sys.exit(f"[orch] {account}: fetch_grmg_ckpts failed rc={rc}")

    # 3. Mapping
    if not args.skip_mapping:
        mapping_local = REPO_ROOT / ".runlogs" / "hf_calvin_mapping.json"
        if not mapping_local.exists():
            sys.exit(f"[orch] mapping JSON missing at {mapping_local}")
        rc = _run_modal_blocking(
            account,
            ["run", "modal_app.py::upload_mapping",
             "--local-path", str(mapping_local)],
            log_path,
        )
        if rc != 0:
            sys.exit(f"[orch] {account}: upload_mapping failed rc={rc}")

    # 4. HF parquet (only really needed for ground_truth account)
    if not args.skip_parquet:
        rc = _run_modal_blocking(
            account, ["run", "modal_app.py::download_hf_parquet"], log_path,
        )
        if rc != 0:
            sys.exit(f"[orch] {account}: download_hf_parquet failed rc={rc}")

    # 5. Per-method eval data
    for method in methods:
        if method == "ground_truth":
            continue
        method_dir = EVALS_ROOT / method
        if not method_dir.exists():
            sys.exit(f"[orch] CALVIN-EVALS/{method} missing at {method_dir}")
        rc = _run_modal_blocking(
            account,
            ["run", "modal_app.py::upload_eval_method",
             "--local-path", str(method_dir),
             "--method-name", method],
            log_path,
        )
        if rc != 0:
            sys.exit(f"[orch] {account}: upload_eval_method({method}) failed rc={rc}")

    print(f"[orch] {account}: bootstrap complete.")


# ---- Launch ----------------------------------------------------------------

def cmd_launch(args):
    """Launch the GPU evaluations for an account, in the background."""
    account = args.account
    methods = args.methods or ACCOUNT_PLAN[account]
    for method in methods:
        is_gt = (method == "ground_truth")
        num_shards = NUM_SHARDS_GT if is_gt else NUM_SHARDS_DEFAULT
        run_args = [
            "run", "modal_app.py::run_method",
            "--method-name", method,
            "--num-shards", str(num_shards),
            "--ep-len", str(args.ep_len),
        ]
        if is_gt:
            run_args += ["--use-hf-parquet-goal"]
        rec = _run_modal_background(account, run_args, tag=f"run_{method}")
        print(f"[orch] launched: pid={rec['pid']} {account}/{method}")


def cmd_launch_all(args):
    """Launch every (account, method) experiment in the background."""
    for account, methods in ACCOUNT_PLAN.items():
        for method in methods:
            is_gt = (method == "ground_truth")
            num_shards = NUM_SHARDS_GT if is_gt else NUM_SHARDS_DEFAULT
            run_args = [
                "run", "modal_app.py::run_method",
                "--method-name", method,
                "--num-shards", str(num_shards),
                "--ep-len", str(args.ep_len),
            ]
            if is_gt:
                run_args += ["--use-hf-parquet-goal"]
            rec = _run_modal_background(account, run_args, tag=f"run_{method}")
            print(f"[orch] launched: pid={rec['pid']} {account}/{method}")
            time.sleep(2.0)  # avoid hammering Modal API on cold starts


def cmd_status(args):
    items = _load_inflight()
    if not items:
        print("[orch] no inflight jobs")
        return
    for it in items:
        pid = it["pid"]
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        marker = "ALIVE" if alive else "DEAD "
        print(f"[orch] {marker}  pid={pid}  {it['profile']}/{it['tag']}  "
              f"started={it['started_at']}  log={it['log']}")
        # Drop dead jobs.
        if not alive:
            _drop_inflight(pid)


def cmd_pull_results(args):
    """Pull per-method shard files from the runs volume into .runlogs/results/<method>/."""
    account = args.account
    methods = args.methods or ACCOUNT_PLAN[account]
    out_root = REPO_ROOT / ".runlogs" / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    for method in methods:
        run_tag = method
        out_dir = out_root / method
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = ORCH_DIR / f"{account}__pull_{method}.log"
        rc = _run_modal_blocking(
            account,
            ["volume", "get", "--force", "rm-policy-runs",
             f"eval_method/{run_tag}", str(out_dir)],
            log_path,
        )
        if rc != 0:
            print(f"[orch] WARN pull failed for {method} from {account}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_bs = sub.add_parser("bootstrap", help="Upload data/ckpts/mapping/parquet/method-data")
    sp_bs.add_argument("--account", required=True)
    sp_bs.add_argument("--methods", nargs="*")
    sp_bs.add_argument("--skip-calvin", action="store_true")
    sp_bs.add_argument("--skip-ckpts", action="store_true")
    sp_bs.add_argument("--skip-mapping", action="store_true")
    sp_bs.add_argument("--skip-parquet", action="store_true")
    sp_bs.set_defaults(func=cmd_bootstrap)

    sp_l = sub.add_parser("launch", help="Background-launch experiments for one account")
    sp_l.add_argument("--account", required=True)
    sp_l.add_argument("--methods", nargs="*")
    sp_l.add_argument("--ep-len", type=int, default=360)
    sp_l.set_defaults(func=cmd_launch)

    sp_la = sub.add_parser("launch_all", help="Background-launch ALL experiments across all accounts")
    sp_la.add_argument("--ep-len", type=int, default=360)
    sp_la.set_defaults(func=cmd_launch_all)

    sp_st = sub.add_parser("status")
    sp_st.set_defaults(func=cmd_status)

    sp_pull = sub.add_parser("pull_results")
    sp_pull.add_argument("--account", required=True)
    sp_pull.add_argument("--methods", nargs="*")
    sp_pull.set_defaults(func=cmd_pull_results)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
