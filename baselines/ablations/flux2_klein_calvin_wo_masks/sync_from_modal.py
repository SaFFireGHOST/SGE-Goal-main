#!/usr/bin/env python
"""
Real-time, standalone Modal -> local sync poller for the i1 ablation
(flux_kontext_calvin_bft_i1).

WHY THIS EXISTS
---------------
The training run is launched detached (`modal run --detach`), so it keeps going on
Modal even after this server's SSH session ends, and Modal auto-restarts it on
preemption (the training function has `retries=` + checkpoint auto-resume). This
poller is a SEPARATE, long-lived process whose only job is to copy fresh data from
the Modal volume down to this machine in (near) real time:

  * every new epoch checkpoint           -> ./models/<MODEL_DIR_NAME>/checkpoint-epoch-N/
  * validation/test visualizations       -> ./models/<MODEL_DIR_NAME>/{validation,test}_visualization/
  * the live training console log         -> ./logs/<LOG_DIR_NAME>/training_console.log
                                             (also copied to ./<EXP_NAME>.log for convenience)
  * tensorboard event files               -> ./logs/<LOG_DIR_NAME>/

It is decoupled from the training stream on purpose, so it survives this client
disconnecting AND Modal preempting/restarting the run.

AUTH (IMPORTANT)
----------------
This NEVER touches the shared global Modal profile (~/.modal.toml). It relies on
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET being set in the environment (launch.sh does
this for the correct account). Both the Modal Python SDK and the `modal` CLI
subprocesses below pick up those env vars and ignore the global profile.

Run forever; stop with Ctrl-C or by killing the PID recorded by launch.sh.
"""

import os
import subprocess
import sys
import time
import shutil

# ============================================================================
# Per-experiment configuration (matches modal_flux_2_klein_train.py)
# ============================================================================
EXP_NAME = "flux_kontext_calvin_bft_i1"
VOLUME_NAME = "flux2-klein-calvin-bft-i1-volume"
MODEL_DIR_NAME = "flux-2-klein-lora-calvin-i1"   # basename of output_dir on the volume
LOG_DIR_NAME = "flux-2-klein-calvin-i1"          # basename of logging_dir on the volume

POLL_INTERVAL_SECONDS = 30

# Remote (volume-relative) paths
REMOTE_MODELS_DIR = f"models/{MODEL_DIR_NAME}"
REMOTE_LOGS_DIR = f"logs/{LOG_DIR_NAME}"

# Local destinations (relative to this file's directory)
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODELS_PARENT = os.path.join(LOCAL_DIR, "models", MODEL_DIR_NAME)
LOCAL_LOGS_PARENT = os.path.join(LOCAL_DIR, "logs")
LOCAL_CONSOLE_LOG = os.path.join(LOCAL_DIR, f"{EXP_NAME}.log")


def _epoch_num(ckpt_name: str) -> int:
    for part in reversed(ckpt_name.split("-")):
        if part.isdigit():
            return int(part)
    return 0


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _modal_get(remote_path: str, local_dest: str) -> bool:
    """`modal volume get --force VOLUME remote_path local_dest` (inherits env auth)."""
    os.makedirs(local_dest, exist_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "modal", "volume", "get", "--force",
                VOLUME_NAME, remote_path, local_dest,
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode == 0:
            return True
        # Not-yet-existing remote paths are normal early in a run; stay quiet about those.
        stderr = (result.stderr or "").strip()
        if "not found" not in stderr.lower() and "no such" not in stderr.lower():
            _log(f"  warn: get {remote_path} failed: {stderr[:300]}")
        return False
    except subprocess.TimeoutExpired:
        _log(f"  warn: get {remote_path} timed out")
        return False
    except Exception as e:
        _log(f"  warn: get {remote_path} error: {e}")
        return False


def _list_remote_checkpoints():
    """Return sorted list of checkpoint dir names on the volume (via SDK listing)."""
    import modal
    vol = modal.Volume.from_name(VOLUME_NAME)
    entries = vol.listdir(REMOTE_MODELS_DIR)
    ckpts = []
    extras = []
    for e in entries:
        name = e.path.rstrip("/").split("/")[-1]
        if name.startswith("checkpoint-epoch-"):
            ckpts.append(name)
        elif name in ("validation_visualization", "test_visualization"):
            extras.append(name)
    ckpts.sort(key=_epoch_num)
    return ckpts, extras


def _checkpoint_is_complete(local_ckpt_path: str) -> bool:
    return os.path.exists(os.path.join(local_ckpt_path, "training_state.json"))


def sync_once():
    # ---- checkpoints (incremental) ----
    try:
        remote_ckpts, extras = _list_remote_checkpoints()
    except Exception as e:
        # Volume / model dir not created yet, or transient listing error.
        msg = str(e)
        if "not found" not in msg.lower():
            _log(f"checkpoint listing not ready yet ({msg[:160]})")
        remote_ckpts, extras = [], []

    if remote_ckpts:
        os.makedirs(LOCAL_MODELS_PARENT, exist_ok=True)
        latest = remote_ckpts[-1]
        new_or_updated = []
        for ckpt in remote_ckpts:
            local_ckpt = os.path.join(LOCAL_MODELS_PARENT, ckpt)
            have_complete = _checkpoint_is_complete(local_ckpt)
            # Always re-fetch the latest (may have been mid-write last cycle); fetch any
            # checkpoint we don't already have complete locally.
            if (ckpt == latest) or (not have_complete):
                if _modal_get(f"{REMOTE_MODELS_DIR}/{ckpt}", LOCAL_MODELS_PARENT):
                    new_or_updated.append(ckpt)
        if new_or_updated:
            _log(f"checkpoints synced: {new_or_updated} "
                 f"(local now has {len(remote_ckpts)} epoch(s) on volume)")

        # ---- visualization dirs (small; refresh each cycle) ----
        for extra in extras:
            _modal_get(f"{REMOTE_MODELS_DIR}/{extra}", LOCAL_MODELS_PARENT)

    # ---- logs (small; refresh each cycle) ----
    if _modal_get(REMOTE_LOGS_DIR, LOCAL_LOGS_PARENT):
        synced_console = os.path.join(LOCAL_LOGS_PARENT, LOG_DIR_NAME, "training_console.log")
        if os.path.exists(synced_console):
            try:
                shutil.copy2(synced_console, LOCAL_CONSOLE_LOG)
            except Exception:
                pass


def main():
    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        _log("WARNING: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set in environment. "
             "This poller would fall back to the shared global profile. Launch it via "
             "launch.sh so it uses the correct account without touching ~/.modal.toml.")
    _log(f"Starting real-time sync poller for {EXP_NAME}")
    _log(f"  volume   : {VOLUME_NAME}")
    _log(f"  models   : volume:{REMOTE_MODELS_DIR}  ->  {LOCAL_MODELS_PARENT}")
    _log(f"  logs     : volume:{REMOTE_LOGS_DIR}  ->  {os.path.join(LOCAL_LOGS_PARENT, LOG_DIR_NAME)}")
    _log(f"  interval : {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            sync_once()
        except KeyboardInterrupt:
            _log("Interrupted; exiting.")
            break
        except Exception as e:
            _log(f"sync cycle error (will retry): {e}")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
