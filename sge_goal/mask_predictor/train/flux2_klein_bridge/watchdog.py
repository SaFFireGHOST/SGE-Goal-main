#!/usr/bin/env python3
"""
Watchdog daemon that monitors whether the Modal training app is running
and automatically restarts it if Modal stops/preempts it.

The restarted training run auto-resumes from the latest checkpoint on the
Modal volume (handled by the training script's auto-resume logic).

Usage:
    python watchdog.py                        # foreground, default 300s interval
    python watchdog.py --interval 180         # custom check interval
    nohup python watchdog.py >> flux2_klein_bridge_fer_mask_prediction.log 2>&1 &
"""

import argparse
import os
import subprocess
import sys
import time

APP_NAME = "flux2-klein-bridge-fer-mask-prediction"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODAL_SCRIPT = os.path.join(SCRIPT_DIR, "modal_flux_2_klein_erm_train.py")
LOCAL_MODELS_DIR = os.path.join(SCRIPT_DIR, "models", "flux-2-klein-erm")
TRAIN_LOG_FILE = os.path.join(LOCAL_MODELS_DIR, "train_output.log")
DEFAULT_CHECK_INTERVAL = 300  # 5 minutes
MAX_RESTARTS = 100
TOTAL_EPOCHS = 75
# If train_output.log doesn't grow for this many seconds while the app is
# "running", consider the training stalled and restart it.
STALL_TIMEOUT = 1200  # 20 minutes

# Modal auth — env-var override so we never depend on the global profile.
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")

# Full path to modal binary — avoids PATH issues in nohup / background shells.
MODAL_BIN = os.environ.get("MODAL_BIN", "modal")


def _modal_env():
    """Return a copy of os.environ with Modal token env vars injected."""
    env = os.environ.copy()
    if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
        env["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
        env["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET
    return env


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def is_training_complete():
    """Check if all training epochs are done locally."""
    if os.path.isdir(os.path.join(LOCAL_MODELS_DIR, "test_visualization")):
        return True
    if os.path.isdir(os.path.join(LOCAL_MODELS_DIR, f"checkpoint-epoch-{TOTAL_EPOCHS}")):
        return True
    return False


def get_latest_local_epoch():
    """Get the highest epoch number from local checkpoints."""
    if not os.path.isdir(LOCAL_MODELS_DIR):
        return 0
    max_epoch = 0
    for d in os.listdir(LOCAL_MODELS_DIR):
        if d.startswith("checkpoint-epoch-") and os.path.isdir(os.path.join(LOCAL_MODELS_DIR, d)):
            try:
                epoch = int(d.split("-")[-1])
                max_epoch = max(max_epoch, epoch)
            except ValueError:
                pass
    return max_epoch


def get_train_log_size():
    """Return the size in bytes of the local train_output.log, or 0."""
    try:
        return os.path.getsize(TRAIN_LOG_FILE)
    except OSError:
        return 0


def _parse_app_lines():
    """Return a list of (app_id, state) for every app matching APP_NAME.

    Uses ``--json`` output for reliable parsing (the table output truncates
    app descriptions, causing false negatives).
    """
    import json as _json

    try:
        result = subprocess.run(
            [MODAL_BIN, "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            env=_modal_env(),
        )
        if result.returncode != 0:
            print(f"[{_timestamp()}] [WATCHDOG] Warning: `modal app list --json` failed (rc={result.returncode})")
            return None  # None = error; caller should assume running

        try:
            apps = _json.loads(result.stdout)
        except _json.JSONDecodeError:
            print(f"[{_timestamp()}] [WATCHDOG] Warning: could not parse JSON from `modal app list --json`")
            return None

        matches = []
        for app in apps:
            desc = (app.get("Description") or "").lower()
            if APP_NAME not in desc:
                continue
            app_id = app.get("App ID")
            state_raw = (app.get("State") or "").lower()
            if any(s in state_raw for s in ("running", "deployed", "starting", "created", "ephemeral", "detached")):
                matches.append((app_id, "running"))
            elif any(s in state_raw for s in ("stopped", "error", "expired")):
                matches.append((app_id, "stopped"))
            else:
                matches.append((app_id, "unknown"))
        return matches

    except subprocess.TimeoutExpired:
        print(f"[{_timestamp()}] [WATCHDOG] Warning: `modal app list` timed out")
        return None
    except FileNotFoundError:
        print(f"[{_timestamp()}] [WATCHDOG] ERROR: 'modal' CLI not found.")
        return None
    except Exception as e:
        print(f"[{_timestamp()}] [WATCHDOG] Error checking app status: {e}")
        return None


def is_app_running():
    """Check if ANY Modal app with our name is currently running.

    Returns True on errors to avoid spurious restarts.
    """
    matches = _parse_app_lines()
    if matches is None:
        return True  # assume running on error
    # Return True if ANY matching app is in a running-like state
    for _app_id, state in matches:
        if state == "running":
            return True
    return False


def kill_apps_by_name():
    """Stop ALL Modal apps matching APP_NAME to prevent duplicates.

    Only stops apps with our specific name — never touches other projects.
    """
    matches = _parse_app_lines()
    if matches is None:
        print(f"[{_timestamp()}] [WATCHDOG] Could not list apps — skipping kill step.")
        return
    if not matches:
        print(f"[{_timestamp()}] [WATCHDOG] No apps matching '{APP_NAME}' found.")
        return

    for app_id, state in matches:
        if app_id is None:
            print(f"[{_timestamp()}] [WATCHDOG] Found matching app but could not parse ID (state={state}). Skipping.")
            continue
        print(f"[{_timestamp()}] [WATCHDOG] Stopping app {app_id} (state={state})...")
        try:
            result = subprocess.run(
                [MODAL_BIN, "app", "stop", app_id],
                capture_output=True,
                text=True,
                timeout=60,
                env=_modal_env(),
            )
            if result.returncode == 0:
                print(f"[{_timestamp()}] [WATCHDOG] App {app_id} stopped successfully.")
            else:
                stderr = result.stderr.strip()[:200] if result.stderr else ""
                print(f"[{_timestamp()}] [WATCHDOG] Warning: stop {app_id} rc={result.returncode} {stderr}")
        except subprocess.TimeoutExpired:
            print(f"[{_timestamp()}] [WATCHDOG] Warning: stop {app_id} timed out.")
        except Exception as e:
            print(f"[{_timestamp()}] [WATCHDOG] Warning: stop {app_id} error: {e}")

    # Brief pause to let Modal clean up
    time.sleep(5)


def restart_training():
    """Restart the training app in detached mode.

    First kills ALL existing apps with our name to prevent duplicates,
    then launches a new training run. The training script's auto-resume
    logic will automatically detect the latest checkpoint on the Modal
    volume and continue from there.
    """
    ts = _timestamp()
    latest_epoch = get_latest_local_epoch()
    print(f"[{ts}] [WATCHDOG] Restarting training (latest local checkpoint: epoch {latest_epoch})...")
    sys.stdout.flush()

    # Step 1: Kill all existing apps with our name to prevent duplicates
    print(f"[{_timestamp()}] [WATCHDOG] Killing any existing apps with name '{APP_NAME}'...")
    kill_apps_by_name()

    # Step 2: Launch new training run
    cmd = [
        MODAL_BIN, "run", "-d",
        MODAL_SCRIPT,
        "--action", "train",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max for the launch (includes upload)
            cwd=SCRIPT_DIR,
            env=_modal_env(),
        )
        ts2 = _timestamp()
        if result.returncode == 0:
            print(f"[{ts2}] [WATCHDOG] Training restarted successfully.")
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines()[-10:]:
                    print(f"  {line}")
        else:
            stderr_snippet = result.stderr.strip()[:500] if result.stderr else "(no stderr)"
            stdout_snippet = result.stdout.strip()[-500:] if result.stdout else ""
            print(f"[{ts2}] [WATCHDOG] Restart failed (rc={result.returncode})")
            if stderr_snippet:
                print(f"  stderr: {stderr_snippet}")
            if stdout_snippet:
                print(f"  stdout: {stdout_snippet}")
    except subprocess.TimeoutExpired:
        print(f"[{_timestamp()}] [WATCHDOG] Restart command timed out after 600s")
    except Exception as e:
        print(f"[{_timestamp()}] [WATCHDOG] Restart error: {e}")

    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Training watchdog daemon")
    parser.add_argument("--interval", type=int, default=DEFAULT_CHECK_INTERVAL,
                        help=f"Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})")
    parser.add_argument("--max-restarts", type=int, default=MAX_RESTARTS,
                        help=f"Maximum number of restarts (default: {MAX_RESTARTS})")
    parser.add_argument("--initial-delay", type=int, default=None,
                        help="Initial delay before first check (default: same as interval)")
    parser.add_argument("--stall-timeout", type=int, default=STALL_TIMEOUT,
                        help=f"Seconds with no new checkpoint before declaring stall (default: {STALL_TIMEOUT})")
    args = parser.parse_args()

    initial_delay = args.initial_delay if args.initial_delay is not None else args.interval
    stall_timeout = args.stall_timeout

    print(f"[{_timestamp()}] [WATCHDOG] Training watchdog started.")
    print(f"[WATCHDOG]   App name:       {APP_NAME}")
    print(f"[WATCHDOG]   Check interval: {args.interval}s")
    print(f"[WATCHDOG]   Max restarts:   {args.max_restarts}")
    print(f"[WATCHDOG]   Initial delay:  {initial_delay}s")
    print(f"[WATCHDOG]   Stall timeout:  {stall_timeout}s")
    print(f"[WATCHDOG]   PID:            {os.getpid()}")
    sys.stdout.flush()

    # Initial delay to let the training app start up
    print(f"[{_timestamp()}] [WATCHDOG] Waiting {initial_delay}s before first check...")
    sys.stdout.flush()
    time.sleep(initial_delay)

    restart_count = 0
    # Stall detection: track the last time train_output.log grew.
    _last_log_size = get_train_log_size()
    _last_activity_time = time.time()

    while restart_count < args.max_restarts:
        ts = _timestamp()
        try:
            if is_training_complete():
                print(f"[{ts}] [WATCHDOG] Training is complete! Exiting watchdog.")
                sys.stdout.flush()
                break

            app_running = is_app_running()
            latest_local = get_latest_local_epoch()
            cur_log_size = get_train_log_size()

            # --- Stall detection ----------------------------------------
            # If the synced train_output.log has grown, training is alive.
            if cur_log_size > _last_log_size:
                _last_log_size = cur_log_size
                _last_activity_time = time.time()

            idle_seconds = time.time() - _last_activity_time
            stalled = app_running and idle_seconds > stall_timeout

            if app_running and not stalled:
                print(
                    f"[{ts}] [WATCHDOG] App is running. "
                    f"Epoch {latest_local}/{TOTAL_EPOCHS}. "
                    f"Log size: {cur_log_size} bytes. "
                    f"Idle: {int(idle_seconds)}s / {stall_timeout}s"
                )
            elif stalled:
                print(
                    f"[{ts}] [WATCHDOG] STALL DETECTED! App is running but "
                    f"train_output.log hasn't grown in {int(idle_seconds)}s "
                    f"(threshold={stall_timeout}s). "
                    f"Epoch {latest_local}/{TOTAL_EPOCHS}. "
                    f"Killing stalled app and restarting..."
                )
                restart_training()
                restart_count += 1
                _last_log_size = get_train_log_size()
                _last_activity_time = time.time()
                print(f"[{_timestamp()}] [WATCHDOG] Restart count: {restart_count}/{args.max_restarts}")
                print(f"[{_timestamp()}] [WATCHDOG] Waiting extra {args.interval}s for app to initialize...")
                sys.stdout.flush()
                time.sleep(args.interval)
            else:
                # App not running
                print(f"[{ts}] [WATCHDOG] App NOT running! Initiating restart #{restart_count + 1}...")
                restart_training()
                restart_count += 1
                _last_log_size = get_train_log_size()
                _last_activity_time = time.time()
                print(f"[{ts}] [WATCHDOG] Restart count: {restart_count}/{args.max_restarts}")
                print(f"[{_timestamp()}] [WATCHDOG] Waiting extra {args.interval}s for app to initialize...")
                sys.stdout.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f"\n[{_timestamp()}] [WATCHDOG] Interrupted. Exiting.")
            break
        except Exception as e:
            print(f"[{ts}] [WATCHDOG] Unexpected error: {e}")

        sys.stdout.flush()
        time.sleep(args.interval)

    if restart_count >= args.max_restarts:
        print(f"[{_timestamp()}] [WATCHDOG] Max restarts ({args.max_restarts}) reached. Exiting.")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
