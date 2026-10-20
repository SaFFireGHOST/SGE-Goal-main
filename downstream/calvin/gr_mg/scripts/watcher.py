#!/usr/bin/env python3
"""Watcher: poll each account's bootstrap_top.log; once complete, launch the
GPU experiments for that account in the background.

A bootstrap is "complete" when its top-log contains the line
``[orch] <account>: bootstrap complete.``
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_DIR = REPO_ROOT / ".runlogs" / "orchestrator"

ACCOUNT_PLAN = {
    "modal-account-2":      ["flux-kontext-dev-ft", "flux2-klein"],
    "modal-account-3":   ["flux2-klein-ft", "gemini-3-pro-image"],
    "modal-account-4": ["gpt-image-1.5", "clover"],
    "modal-account-5":     ["qwen-image-edit", "susie"],
    "modal-account-6":       ["ground_truth"],
}
NUM_SHARDS_GT = 10
NUM_SHARDS_DEFAULT = 5


def is_bootstrap_done(account: str) -> bool:
    p = ORCH_DIR / f"{account}__bootstrap_top.log"
    if not p.exists():
        return False
    try:
        text = p.read_text()
    except Exception:
        return False
    return f"[orch] {account}: bootstrap complete." in text


def already_launched(account: str, method: str) -> bool:
    return (ORCH_DIR / f"{account}__run_{method}.LAUNCHED").exists()


def already_skeletons_uploaded(account: str) -> bool:
    return (ORCH_DIR / f"{account}__skeletons_uploaded.OK").exists()


def upload_skeletons(account: str) -> bool:
    """Upload chain_skeletons.json to the account's runs volume."""
    sentinel = ORCH_DIR / f"{account}__skeletons_uploaded.OK"
    log_path = ORCH_DIR / f"{account}__skeletons_upload.log"
    cmd = ["modal", "run", "modal_app.py::upload_chain_skeletons"]
    env = os.environ.copy()
    env["MODAL_PROFILE"] = account
    print(f"[watcher] uploading chain_skeletons to {account}")
    with log_path.open("a") as logf:
        logf.write(f"\n=== {datetime.utcnow().isoformat()}Z $ {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
        )
    if proc.returncode == 0:
        sentinel.touch()
        return True
    print(f"[watcher] WARN: chain_skeletons upload failed for {account}")
    return False


def launch(account: str, method: str) -> None:
    is_gt = method == "ground_truth"
    num_shards = NUM_SHARDS_GT if is_gt else NUM_SHARDS_DEFAULT
    log_path = ORCH_DIR / f"{account}__run_{method}.log"
    sentinel = ORCH_DIR / f"{account}__run_{method}.LAUNCHED"

    cmd = [
        "modal", "run", "--detach", "modal_app.py::run_method",
        "--method-name", method,
        "--num-shards", str(num_shards),
        "--ep-len", "360",
    ]
    if is_gt:
        cmd += ["--use-hf-parquet-goal"]

    print(f"[watcher] launching {account}/{method}")
    env = os.environ.copy()
    env["MODAL_PROFILE"] = account
    with log_path.open("a") as logf:
        logf.write(
            f"\n\n=== {datetime.utcnow().isoformat()}Z BG $ {' '.join(cmd)}\n"
        )
        logf.flush()
        subprocess.Popen(
            cmd, env=env, cwd=str(REPO_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    sentinel.touch()


def main():
    print(f"[watcher] starting; tracking {len(ACCOUNT_PLAN)} accounts")
    while True:
        all_done = True
        for account, methods in ACCOUNT_PLAN.items():
            if is_bootstrap_done(account):
                if not already_skeletons_uploaded(account):
                    upload_skeletons(account)
                for method in methods:
                    if not already_launched(account, method):
                        launch(account, method)
                        time.sleep(2)
            else:
                all_done = False
        if all(is_bootstrap_done(a) for a in ACCOUNT_PLAN) and all(
            already_launched(a, m)
            for a, ms in ACCOUNT_PLAN.items()
            for m in ms
        ):
            print("[watcher] every account/method launched; exiting")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
