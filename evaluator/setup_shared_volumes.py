#!/usr/bin/env python3
"""
setup_shared_volumes.py — Populate the shared lgie-eval-weights and lgie-eval-data
volumes on fresh Modal accounts.

These volumes are shared across ALL eval runs on a given account:
  - lgie-eval-weights: SAM (2.4GB), LoFTR (177MB), XMem (475MB)
  - lgie-eval-data:    original scene images (569MB)

HuggingFace model weights (BLIP-2, VideoChat-Flash, Qwen2.5-7B) will auto-download
from HF hub at runtime since HF_HOME points to the weights volume.

Usage:
    python setup_shared_volumes.py --account C          # Setup Account C only
    python setup_shared_volumes.py --account D          # Setup Account D only
    python setup_shared_volumes.py --account all        # Setup both
    python setup_shared_volumes.py --account C --dry-run
    python setup_shared_volumes.py --account C --skip-weights  # Only upload eval-data
    python setup_shared_volumes.py --account C --skip-data     # Only upload weights
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODAL = os.environ.get("MODAL_BIN", "modal")

# ── Account credentials ──
ACCOUNT_C = {
    "name": "account-c",
    "token_id": os.environ.get("MODAL_TOKEN_ID", ""),
    "token_secret": os.environ.get("MODAL_TOKEN_SECRET", ""),
}
ACCOUNT_D = {
    "name": "account-d",
    "token_id": os.environ.get("MODAL_TOKEN_ID", ""),
    "token_secret": os.environ.get("MODAL_TOKEN_SECRET", ""),
}

# ── Shared volume names (hardcoded in modal_lgie_eval.py) ──
WEIGHTS_VOLUME = "lgie-eval-weights"
EVAL_DATA_VOLUME = "lgie-eval-data"

# ── Local paths to weight files ──
SAM_WEIGHTS = SCRIPT_DIR / "sam_vit_h_4b8939.pth"          # 2.4 GB
LOFTR_WEIGHTS_DIR = SCRIPT_DIR / "loftr_weights"            # 177 MB (4 .ckpt files)
XMEM_WEIGHTS_DIR = SCRIPT_DIR / "XMem" / "saves"           # 475 MB (2 .pth files)
EVAL_DATA_DIR = SCRIPT_DIR / "lgie-eval-data"               # 569 MB


def _make_env(account):
    """Return env dict with Modal credentials."""
    env = os.environ.copy()
    if account["token_id"] and account["token_secret"]:
        env["MODAL_TOKEN_ID"] = account["token_id"]
        env["MODAL_TOKEN_SECRET"] = account["token_secret"]
    return env


def _run_modal(args, account, desc, timeout=3600):
    """Run a modal CLI command, printing output."""
    env = _make_env(account)
    cmd = [MODAL] + args
    print(f"    $ {' '.join(args)}")
    sys.stdout.flush()

    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=timeout,
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            print(f"      {line}")
    if result.returncode != 0:
        print(f"    ✗ FAILED: {desc}")
        if result.stderr.strip():
            for line in result.stderr.strip().split("\n"):
                print(f"      [err] {line}")
        return False
    return True


def create_volume_if_needed(vol_name, account, dry_run=False):
    """Create a Modal volume if it doesn't exist."""
    if dry_run:
        print(f"  [DRY] Would create volume '{vol_name}'")
        return True

    # Try listing first — if the volume exists, skip creation
    env = _make_env(account)
    result = subprocess.run(
        [MODAL, "volume", "list"], env=env, capture_output=True, text=True, timeout=30,
    )
    if vol_name in result.stdout:
        print(f"  ✓ Volume '{vol_name}' already exists")
        return True

    print(f"  Creating volume '{vol_name}'...")
    return _run_modal(["volume", "create", vol_name], account, f"create {vol_name}")


def upload_weights(account, acct_label, dry_run=False):
    """Upload SAM, LoFTR, and XMem weights to the shared weights volume."""
    print(f"\n  [WEIGHTS] Uploading model weights to '{WEIGHTS_VOLUME}'...")

    # Verify local files exist
    missing = []
    if not SAM_WEIGHTS.exists():
        missing.append(f"SAM: {SAM_WEIGHTS}")
    if not LOFTR_WEIGHTS_DIR.exists():
        missing.append(f"LoFTR: {LOFTR_WEIGHTS_DIR}")
    if not XMEM_WEIGHTS_DIR.exists():
        missing.append(f"XMem: {XMEM_WEIGHTS_DIR}")

    if missing:
        print(f"  ✗ Missing local weight files:")
        for m in missing:
            print(f"      {m}")
        return False

    if dry_run:
        print(f"  [DRY] Would upload SAM weights ({SAM_WEIGHTS.stat().st_size / 1e9:.1f} GB)")
        print(f"  [DRY] Would upload LoFTR weights dir ({LOFTR_WEIGHTS_DIR})")
        print(f"  [DRY] Would upload XMem weights dir ({XMEM_WEIGHTS_DIR})")
        return True

    # SAM weights → /sam_vit_h_4b8939.pth
    print(f"    Uploading SAM weights (2.4 GB)...")
    t0 = time.time()
    ok = _run_modal(
        ["volume", "put", WEIGHTS_VOLUME, str(SAM_WEIGHTS), "/sam_vit_h_4b8939.pth"],
        account, "upload SAM weights", timeout=1800,
    )
    if not ok:
        return False
    print(f"    SAM uploaded in {time.time() - t0:.0f}s")

    # LoFTR weights → /loftr_weights/
    print(f"    Uploading LoFTR weights (177 MB)...")
    t0 = time.time()
    ok = _run_modal(
        ["volume", "put", WEIGHTS_VOLUME, str(LOFTR_WEIGHTS_DIR), "/loftr_weights"],
        account, "upload LoFTR weights", timeout=600,
    )
    if not ok:
        return False
    print(f"    LoFTR uploaded in {time.time() - t0:.0f}s")

    # XMem weights → /XMem/
    print(f"    Uploading XMem weights (475 MB)...")
    t0 = time.time()
    ok = _run_modal(
        ["volume", "put", WEIGHTS_VOLUME, str(XMEM_WEIGHTS_DIR), "/XMem"],
        account, "upload XMem weights", timeout=900,
    )
    if not ok:
        return False
    print(f"    XMem uploaded in {time.time() - t0:.0f}s")

    print(f"  ✓ All weights uploaded to '{WEIGHTS_VOLUME}' on Account {acct_label}")
    return True


def upload_eval_data(account, acct_label, dry_run=False):
    """Upload lgie-eval-data (original scene images) to the shared eval-data volume."""
    print(f"\n  [EVAL-DATA] Uploading original scene images to '{EVAL_DATA_VOLUME}'...")

    if not EVAL_DATA_DIR.exists():
        print(f"  ✗ Missing local eval-data dir: {EVAL_DATA_DIR}")
        return False

    if dry_run:
        print(f"  [DRY] Would upload eval-data dir ({EVAL_DATA_DIR})")
        return True

    # Upload entire lgie-eval-data/ → /lgie-eval-data/
    print(f"    Uploading eval-data (569 MB)...")
    t0 = time.time()
    ok = _run_modal(
        ["volume", "put", EVAL_DATA_VOLUME, str(EVAL_DATA_DIR), "/lgie-eval-data"],
        account, "upload eval-data", timeout=1200,
    )
    if not ok:
        return False
    print(f"    Eval-data uploaded in {time.time() - t0:.0f}s")

    print(f"  ✓ Eval-data uploaded to '{EVAL_DATA_VOLUME}' on Account {acct_label}")
    return True


def verify_volumes(account, acct_label):
    """Quick verification that volumes have content."""
    print(f"\n  [VERIFY] Checking volumes on Account {acct_label}...")
    env = _make_env(account)

    # Check weights volume
    result = subprocess.run(
        [MODAL, "volume", "ls", WEIGHTS_VOLUME, "/"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        print(f"    {WEIGHTS_VOLUME}: {len(lines)} entries at root")
        for line in lines:
            print(f"      {line}")
    else:
        print(f"    ✗ Could not list {WEIGHTS_VOLUME}")

    # Check eval-data volume
    result = subprocess.run(
        [MODAL, "volume", "ls", EVAL_DATA_VOLUME, "/"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        print(f"    {EVAL_DATA_VOLUME}: {len(lines)} entries at root")
        for line in lines:
            print(f"      {line}")
    else:
        print(f"    ✗ Could not list {EVAL_DATA_VOLUME}")


def setup_account(account, acct_label, dry_run=False, skip_weights=False, skip_data=False):
    """Full setup for one account: create volumes + upload weights + eval-data."""
    print(f"\n{'='*64}")
    print(f"Setting up Account {acct_label} ({account['name']})")
    print(f"{'='*64}")

    # Step 1: Create volumes
    if not skip_weights:
        if not create_volume_if_needed(WEIGHTS_VOLUME, account, dry_run):
            return False

    if not skip_data:
        if not create_volume_if_needed(EVAL_DATA_VOLUME, account, dry_run):
            return False

    # Step 2: Upload weights
    if not skip_weights:
        if not upload_weights(account, acct_label, dry_run):
            return False

    # Step 3: Upload eval-data
    if not skip_data:
        if not upload_eval_data(account, acct_label, dry_run):
            return False

    # Step 4: Verify
    if not dry_run:
        verify_volumes(account, acct_label)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup shared volumes (weights + eval-data) on fresh Modal accounts"
    )
    parser.add_argument("--account", required=True, choices=["C", "D", "all"],
                        help="Which account(s) to setup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    parser.add_argument("--skip-weights", action="store_true",
                        help="Skip uploading model weights")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip uploading eval-data")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing volumes (no upload)")
    args = parser.parse_args()

    accounts = []
    if args.account in ("C", "all"):
        accounts.append(("C", ACCOUNT_C))
    if args.account in ("D", "all"):
        accounts.append(("D", ACCOUNT_D))

    print(f"{'='*64}")
    print(f"SHARED VOLUME SETUP — {__import__('datetime').datetime.now()}")
    print(f"{'='*64}")

    if args.verify_only:
        for label, account in accounts:
            verify_volumes(account, label)
        return

    all_ok = True
    for label, account in accounts:
        ok = setup_account(
            account, label,
            dry_run=args.dry_run,
            skip_weights=args.skip_weights,
            skip_data=args.skip_data,
        )
        if not ok:
            all_ok = False
            print(f"\n⚠ Account {label} setup FAILED")

    if all_ok:
        print(f"\n{'='*64}")
        if args.dry_run:
            print("DRY RUN complete.")
        else:
            print("All shared volumes set up successfully!")
            print("You can now run: python resume_all.py")
        print(f"{'='*64}")
    else:
        print(f"\n{'='*64}")
        print("Some setups failed — check errors above.")
        print(f"{'='*64}")
        sys.exit(1)


if __name__ == "__main__":
    main()
