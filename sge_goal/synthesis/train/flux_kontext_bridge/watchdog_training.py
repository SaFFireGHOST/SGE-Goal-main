#!/usr/bin/env python3
"""
Watchdog process that monitors Modal training app and restarts if stopped/stalled.

Monitors via two mechanisms:
1. Check running apps via `modal app list` CLI
2. Check heartbeat freshness in the signal log on the Modal volume

If training is detected as stopped (app not running) or stalled (no heartbeat
for a long time), the watchdog:
1. Kills all apps matching our app name (prevents duplicates)
2. Relaunches training via `modal run -d --launch-only`
3. Restarts log streaming

Usage:
    python watchdog_training.py [--check-interval SECONDS] [--grace-period SECONDS]
"""

import argparse
import atexit
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import modal

# Must match modal_flux_kontext_train.py
APP_NAME = "flux-kontext-dev-bridge-image-editing"
VOLUME_NAME = "flux-kontext-dev-bridge-image-editing-volume"
SIGNAL_LOG_VOLUME_PATH = ".training_signals.jsonl"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODAL_SCRIPT = os.path.join(SCRIPT_DIR, "modal_flux_kontext_train.py")
LOG_FILE = os.path.join(SCRIPT_DIR, "flux_kontext_dev_bridge_image_editing.log")
PID_FILE = os.path.join(SCRIPT_DIR, ".watchdog.pid")

# Resolve modal CLI path — nohup doesn't inherit conda PATH
MODAL_CLI = os.path.join(os.path.dirname(sys.executable), "modal")

DEFAULT_CHECK_INTERVAL = 120   # seconds between checks
DEFAULT_GRACE_PERIOD = 600     # seconds after restart before checking again
HEARTBEAT_STALE_THRESHOLD = 900  # 15 min: if no heartbeat for this long, training is stalled

# Graceful shutdown
_shutdown = False
_log_streamer_proc = None


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    log("Received shutdown signal.")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGHUP, signal.SIG_IGN)  # Ignore terminal hangup


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WATCHDOG {ts}] {msg}", flush=True)


def _acquire_pid_lock():
    """Ensure only one watchdog runs. Kill stale PID if needed."""
    if os.path.exists(PID_FILE):
        try:
            old_pid = int(open(PID_FILE).read().strip())
            # Check if that process is still alive
            os.kill(old_pid, 0)
            # It's alive — is it actually a watchdog?
            try:
                cmdline = open(f"/proc/{old_pid}/cmdline", "rb").read().decode(errors="replace")
                if "watchdog_training" in cmdline:
                    log(f"Another watchdog is already running (PID {old_pid}). Exiting.")
                    sys.exit(0)
            except (FileNotFoundError, PermissionError):
                pass
        except (OSError, ValueError):
            # Process not running or bad PID file — stale, remove it
            pass
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(_release_pid_lock)
    log(f"PID lock acquired (PID {os.getpid()})")


def _release_pid_lock():
    try:
        if os.path.exists(PID_FILE):
            pid = int(open(PID_FILE).read().strip())
            if pid == os.getpid():
                os.remove(PID_FILE)
    except Exception:
        pass


def _get_app_list_json():
    """Get modal app list as JSON (avoids truncated table output)."""
    try:
        result = subprocess.run(
            [MODAL_CLI, "app", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            log(f"`modal app list --json` failed (rc={result.returncode}): {result.stderr.strip()}")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        log("`modal app list --json` timed out.")
        return None
    except (json.JSONDecodeError, Exception) as e:
        log(f"Error parsing app list: {type(e).__name__}: {e}")
        return None


def is_app_running():
    """
    Check if our Modal app is currently running via `modal app list --json`.
    Returns True if running, False if not, None if check failed.
    """
    apps = _get_app_list_json()
    if apps is None:
        return None
    for app in apps:
        desc = app.get("Description", "")
        state = app.get("State", "").lower()
        if desc == APP_NAME and app.get("Stopped at") is None:
            if any(s in state for s in ("running", "deployed", "ephemeral", "detached")):
                return True
    return False


def get_our_app_id():
    """
    Find the app ID for our running app from `modal app list --json`.
    Returns the app ID string (e.g. 'ap-xxx') or None.
    """
    apps = _get_app_list_json()
    if apps is None:
        return None
    for app in apps:
        desc = app.get("Description", "")
        if desc == APP_NAME and app.get("Stopped at") is None:
            return app.get("App ID")
    return None


def get_last_heartbeat_age():
    """
    Read the signal log from the Modal volume and return how many seconds
    ago the last heartbeat (or any signal) was written.
    Returns None on error or if no signals exist.
    """
    try:
        volume = modal.Volume.from_name(VOLUME_NAME)
        buf = io.BytesIO()
        volume.read_file_into_fileobj(SIGNAL_LOG_VOLUME_PATH, buf)
        buf.seek(0)
        content = buf.read().decode("utf-8").strip()
        if not content:
            return None

        # Get the last signal entry
        last_line = None
        for line in content.split("\n"):
            line = line.strip()
            if line:
                last_line = line

        if not last_line:
            return None

        entry = json.loads(last_line)
        ts_str = entry.get("timestamp")
        if not ts_str:
            return None

        last_ts = datetime.fromisoformat(ts_str)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - last_ts).total_seconds()
    except Exception as e:
        log("Could not read heartbeat from volume: {0}: {1}".format(type(e).__name__, e))
        return None


def stop_all_matching_apps():
    """Stop all running Modal apps matching our app name."""
    apps = _get_app_list_json()
    if apps is None:
        return
    for app in apps:
        desc = app.get("Description", "")
        if desc == APP_NAME and app.get("Stopped at") is None:
            app_id = app.get("App ID")
            if app_id:
                log(f"Stopping app: {app_id}")
                try:
                    subprocess.run(
                        [MODAL_CLI, "app", "stop", app_id],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                except Exception as e:
                    log(f"Error stopping {app_id}: {e}")


def start_log_streaming():
    """Start `modal app logs` in the background, appending to log file."""
    global _log_streamer_proc
    stop_log_streaming()

    app_id = get_our_app_id()
    if not app_id:
        log("Could not find app ID for log streaming. Will retry on next cycle.")
        return

    try:
        log_fh = open(LOG_FILE, "a")
        _log_streamer_proc = subprocess.Popen(
            [MODAL_CLI, "app", "logs", app_id],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        log("Log streamer started for {0} (PID: {1})".format(app_id, _log_streamer_proc.pid))
    except Exception as e:
        log("Failed to start log streaming: " + str(e))


def stop_log_streaming():
    """Stop the current log streaming process if running."""
    global _log_streamer_proc
    if _log_streamer_proc is not None:
        try:
            _log_streamer_proc.terminate()
            _log_streamer_proc.wait(timeout=10)
        except Exception:
            try:
                _log_streamer_proc.kill()
            except Exception:
                pass
        _log_streamer_proc = None


def launch_training():
    """Launch training via modal run -d --launch-only."""
    log("Launching training in detached mode...")
    cmd = [
        MODAL_CLI,
        "run",
        "-d",
        MODAL_SCRIPT,
        "--action",
        "train",
        "--single-gpu",
        "--launch-only",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=SCRIPT_DIR,
        )
        if result.returncode == 0:
            log("Training launched successfully.")
            for line in (result.stdout + result.stderr).strip().split("\n")[-5:]:
                if line.strip():
                    log("  " + line.strip())
        else:
            log("Failed to launch training (rc={0}):".format(result.returncode))
            for line in (result.stderr + result.stdout).strip().split("\n")[-10:]:
                if line.strip():
                    log("  " + line.strip())
    except subprocess.TimeoutExpired:
        log("Training launch timed out after 600s.")
    except Exception as e:
        log("Error launching training: {0}: {1}".format(type(e).__name__, e))


def do_restart():
    """Full restart cycle: stop old apps, launch new training, start log streaming."""
    log("=== RESTARTING TRAINING ===")
    stop_all_matching_apps()
    time.sleep(15)
    launch_training()
    time.sleep(10)
    start_log_streaming()
    log("=== RESTART COMPLETE ===")


def main():
    parser = argparse.ArgumentParser(description="Watchdog for Modal training")
    parser.add_argument(
        "--check-interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
    )
    parser.add_argument(
        "--grace-period",
        type=int,
        default=DEFAULT_GRACE_PERIOD,
    )
    args = parser.parse_args()

    # Prevent duplicate watchdog processes
    _acquire_pid_lock()

    log("Starting watchdog (check_interval={0}s, grace_period={1}s)".format(
        args.check_interval, args.grace_period))
    log("Monitoring app: " + APP_NAME)
    log("Heartbeat stale threshold: {0}s".format(HEARTBEAT_STALE_THRESHOLD))

    # Start log streaming for the current run
    start_log_streaming()

    # Initial grace period
    log("Initial grace period of {0}s...".format(args.grace_period))
    for _ in range(args.grace_period):
        if _shutdown:
            break
        time.sleep(1)

    consecutive_not_running = 0
    required_not_running = 3

    while not _shutdown:
        app_running = is_app_running()

        if app_running is True:
            consecutive_not_running = 0
            hb_age = get_last_heartbeat_age()
            if hb_age is not None and hb_age > HEARTBEAT_STALE_THRESHOLD:
                log("WARNING: App is running but last heartbeat was {0:.0f}s ago "
                    "(threshold: {1}s). Training may be stalled.".format(
                        hb_age, HEARTBEAT_STALE_THRESHOLD))
                stall_confirmed = True
                for _ in range(2):
                    time.sleep(args.check_interval)
                    if _shutdown:
                        break
                    hb_age2 = get_last_heartbeat_age()
                    if hb_age2 is not None and hb_age2 <= HEARTBEAT_STALE_THRESHOLD:
                        stall_confirmed = False
                        break

                if stall_confirmed and not _shutdown:
                    log("Training confirmed stalled. Restarting...")
                    do_restart()
                    for _ in range(args.grace_period):
                        if _shutdown:
                            break
                        time.sleep(1)
                    continue
            else:
                if hb_age is not None:
                    log("Training is running (heartbeat age: {0:.0f}s).".format(hb_age))
                else:
                    log("Training is running (heartbeat not available yet).")

        elif app_running is False:
            consecutive_not_running += 1
            log("Training NOT running ({0}/{1} checks).".format(
                consecutive_not_running, required_not_running))

            if consecutive_not_running >= required_not_running:
                # Check if training actually completed
                try:
                    volume = modal.Volume.from_name(VOLUME_NAME)
                    buf = io.BytesIO()
                    volume.read_file_into_fileobj(SIGNAL_LOG_VOLUME_PATH, buf)
                    content = buf.getvalue().decode("utf-8").strip()
                    last_line = content.split("\n")[-1] if content else ""
                    if last_line:
                        last_signal = json.loads(last_line)
                        if last_signal.get("signal") == "training_complete":
                            log("Training completed successfully. Watchdog exiting.")
                            stop_log_streaming()
                            return
                except Exception:
                    pass

                log("Training appears stopped without completing. Restarting...")
                do_restart()
                consecutive_not_running = 0
                for _ in range(args.grace_period):
                    if _shutdown:
                        break
                    time.sleep(1)
                continue
        else:
            log("App status check inconclusive. Skipping this cycle.")

        # Ensure log streamer is alive
        if _log_streamer_proc is None or _log_streamer_proc.poll() is not None:
            start_log_streaming()

        # Sleep between checks
        for _ in range(args.check_interval):
            if _shutdown:
                break
            time.sleep(1)

    stop_log_streaming()
    log("Watchdog shutting down.")


if __name__ == "__main__":
    main()
