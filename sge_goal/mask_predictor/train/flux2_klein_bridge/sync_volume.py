#!/usr/bin/env python3
"""
Background sync daemon that continuously downloads model checkpoints and
visualizations from the Modal volume to the local machine.

Runs in an infinite loop using `modal volume get` CLI (subprocess) to
avoid the Modal Python SDK's high memory consumption that causes OOM and
stalls after extended (~10 hr) runs.

Usage:
    python sync_volume.py                      # foreground, default 180s interval
    python sync_volume.py --interval 120       # custom interval
    nohup python sync_volume.py >> flux2_klein_bridge_fer_mask_prediction.log 2>&1 &
"""

import argparse
import hashlib
import json as _json
import os
import re
import subprocess
import sys
import threading
import time

VOLUME_NAME = "flux2-klein-bridge-fer-mask-prediction-volume"
VOLUME_PATH = "models/flux-2-klein-erm"
APP_NAME = "flux2-klein-bridge-fer-mask-prediction"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DEST = os.path.join(SCRIPT_DIR, "models", "flux-2-klein-erm")
MAIN_LOG_FILE = os.path.join(SCRIPT_DIR, "flux2_klein_bridge_fer_mask_prediction.log")
DEFAULT_INTERVAL = 180  # 3 minutes
SYNC_TIMEOUT = 1800     # 30 minutes max per sync attempt
TOTAL_EPOCHS = 75

# Modal auth — env-var override so we never depend on the global profile.
# The CLI honours MODAL_TOKEN_ID / MODAL_TOKEN_SECRET if present.
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
    """Check if all training epochs are done (final test visualization exists)."""
    if os.path.isdir(os.path.join(LOCAL_DEST, "test_visualization")):
        return True
    if os.path.isdir(os.path.join(LOCAL_DEST, f"checkpoint-epoch-{TOTAL_EPOCHS}")):
        return True
    return False


def sync_once():
    """Run one sync pass: download volume contents to local.

    ``modal volume get`` with a directory source only fetches sub-directories
    reliably — loose files at the directory root (like ``train_output.log``)
    are silently skipped in some CLI versions.  We therefore run a second,
    explicit download for every known loose file that must be synced.
    """
    os.makedirs(LOCAL_DEST, exist_ok=True)
    ts = _timestamp()
    print(f"[{ts}] [SYNC] Starting volume sync: {VOLUME_NAME}:/{VOLUME_PATH} -> {LOCAL_DEST}")
    sys.stdout.flush()

    # --- 1. Bulk sync the whole directory tree (checkpoints, viz dirs) ----
    # ``modal volume get VOL remote_dir local_dir`` places the contents
    # into ``local_dir/<basename(remote_dir)>/``, so we pass the *parent*
    # of LOCAL_DEST so the final path is LOCAL_DEST itself.
    local_parent = os.path.dirname(LOCAL_DEST)
    cmd = [
        MODAL_BIN, "volume", "get",
        VOLUME_NAME,
        VOLUME_PATH,
        local_parent,
        "--force",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SYNC_TIMEOUT,
            env=_modal_env(),
        )
        ts2 = _timestamp()
        if result.returncode == 0:
            # Count local checkpoints for summary
            ckpts = sorted(
                [d for d in os.listdir(LOCAL_DEST)
                 if d.startswith("checkpoint") and os.path.isdir(os.path.join(LOCAL_DEST, d))],
                key=lambda d: int(re.search(r"(\d+)$", d).group(1)) if re.search(r"(\d+)$", d) else 0,
            ) if os.path.isdir(LOCAL_DEST) else []
            latest = ckpts[-1] if ckpts else "none"
            has_val_viz = os.path.isdir(os.path.join(LOCAL_DEST, "validation_visualization"))
            has_test_viz = os.path.isdir(os.path.join(LOCAL_DEST, "test_visualization"))
            print(
                f"[{ts2}] [SYNC] Sync complete. "
                f"Checkpoints: {len(ckpts)} (latest: {latest}) | "
                f"val_viz: {has_val_viz} | test_viz: {has_test_viz}"
            )
        else:
            stderr_snippet = result.stderr.strip()[:500] if result.stderr else "(no stderr)"
            print(f"[{ts2}] [SYNC] Sync failed (rc={result.returncode}): {stderr_snippet}")
    except subprocess.TimeoutExpired:
        print(f"[{_timestamp()}] [SYNC] Sync timed out after {SYNC_TIMEOUT}s — will retry next cycle.")
    except FileNotFoundError:
        print(f"[{_timestamp()}] [SYNC] ERROR: 'modal' CLI not found. Ensure Modal is installed.")
    except Exception as e:
        print(f"[{_timestamp()}] [SYNC] Unexpected error: {e}")

    # --- 2. Explicitly fetch loose files that bulk-dir sync misses --------
    _sync_loose_file("train_output.log")

    sys.stdout.flush()


def _sync_loose_file(filename):
    """Download a single file from the volume root into LOCAL_DEST."""
    remote_path = f"{VOLUME_PATH}/{filename}"
    local_path = os.path.join(LOCAL_DEST, filename)
    try:
        result = subprocess.run(
            [MODAL_BIN, "volume", "get", VOLUME_NAME, remote_path, local_path, "--force"],
            capture_output=True, text=True, timeout=120, env=_modal_env(),
        )
        if result.returncode != 0:
            # File may not exist yet on the volume — not an error
            pass
    except Exception:
        pass

    sys.stdout.flush()


# ── Noise filter regex (shared by LogStreamer) ────────────────────────────
_SKIP_RE = re.compile(
    r"Training:\s+\d+%\|(?!.*flow=)"   # tqdm bar WITHOUT loss info after it
    r"|^Fetching \d+ files:"
    r"|^Loading weights:"
    r"|^Loading checkpoint shards:"
    r"|^Downloading data:"
    r"|^Generating .* split:"
    r"|HTTP Request:"
    r"|huggingface\.co/.*/resolve/"
    r"|^\s*warnings\.warn\("
    r"|local_dir_use_symlinks"
    r"|^\s*$"
)

_GLOBAL_STEP_RE = re.compile(r"\|\s*(\d+)/\d+\s+\[")

# Strip tqdm prefix for dedup: everything up to and including "] " or "]" before
# a timestamp is noise that can differ between sources.
_TQDM_PREFIX_RE = re.compile(r"^Training:\s+\d+%\|.*?\]\s*")


class LogStreamer:
    """Stream ``modal app logs`` in real-time, filter noise, and append
    meaningful lines to *MAIN_LOG_FILE*.

    Runs ``modal app logs <app-id>`` as a background subprocess read by a
    daemon thread.  Deduplicates on restart via a set of already-written
    line hashes so that the historical replay produced by ``modal app logs``
    never re-appends content that's already in the main log.
    """

    def __init__(self, min_global_step: int = 0):
        self._min_global_step = min_global_step
        self._proc = None
        self._thread = None
        self._app_id = None
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._backfilled = False
        self._scan_existing_log()

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_key(line: str) -> str:
        """Normalise *line* for dedup: strip tqdm prefix + whitespace."""
        core = _TQDM_PREFIX_RE.sub("", line).strip()
        return hashlib.md5(core.encode()).hexdigest()

    def _scan_existing_log(self):
        """Pre-populate *_seen* from what's already in the main log so
        restarts don't produce duplicates."""
        try:
            with open(MAIN_LOG_FILE, "r", errors="replace") as f:
                for line in f:
                    self._seen.add(self._dedup_key(line))
        except FileNotFoundError:
            pass

    def _backfill_from_volume_log(self):
        """One-time backfill from the volume's train_output.log.

        ``modal app logs`` only retains a limited recent window, so
        historical entries (especially from early in the run) are only
        available in the volume file.  We read it, filter, dedup
        against *_seen*, and append anything new to *MAIN_LOG_FILE*.
        """
        vol_log = os.path.join(LOCAL_DEST, "train_output.log")
        if not os.path.isfile(vol_log):
            return
        new_lines = []
        with open(vol_log, "r", errors="replace") as f:
            for line in f:
                if not self._is_meaningful(line):
                    continue
                m = _GLOBAL_STEP_RE.search(line)
                if m and int(m.group(1)) < self._min_global_step:
                    continue
                h = self._dedup_key(line)
                with self._lock:
                    if h in self._seen:
                        continue
                    self._seen.add(h)
                new_lines.append(line if line.endswith("\n") else line + "\n")
        if new_lines:
            with open(MAIN_LOG_FILE, "a") as f:
                f.writelines(new_lines)
            print(f"[{_timestamp()}] [SYNC] Backfilled {len(new_lines)} lines from volume log")
            sys.stdout.flush()

    @staticmethod
    def _is_meaningful(line: str) -> bool:
        stripped = line.rstrip("\n\r")
        if not stripped:
            return False
        return not _SKIP_RE.search(stripped)

    def _find_running_app_id(self):
        try:
            result = subprocess.run(
                [MODAL_BIN, "app", "list", "--json"],
                capture_output=True, text=True, timeout=60, env=_modal_env(),
            )
            if result.returncode != 0:
                return None
            apps = _json.loads(result.stdout)
            for app in apps:
                desc = (app.get("Description") or "").lower()
                state = (app.get("State") or "").lower()
                if APP_NAME in desc and any(
                    s in state for s in ("running", "ephemeral", "detached")
                ):
                    return app.get("App ID")
        except Exception:
            pass
        return None

    # ── background reader ────────────────────────────────────────────────

    def _reader(self):
        """Read from the Popen stdout, filter, dedup, and append."""
        try:
            for raw_line in self._proc.stdout:
                if not self._is_meaningful(raw_line):
                    continue
                # Global-step filter
                m = _GLOBAL_STEP_RE.search(raw_line)
                if m and int(m.group(1)) < self._min_global_step:
                    continue
                # Dedup
                h = self._dedup_key(raw_line)
                with self._lock:
                    if h in self._seen:
                        continue
                    self._seen.add(h)
                # Append
                with open(MAIN_LOG_FILE, "a") as f:
                    f.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
        except Exception:
            pass  # process terminated — handled by ensure_running

    # ── public API ───────────────────────────────────────────────────────

    def ensure_running(self):
        """Start or reconnect the log streamer.  Safe to call repeatedly."""
        # One-time volume-based backfill (historical entries modal app logs lost)
        if not self._backfilled:
            self._backfill_from_volume_log()
            self._backfilled = True

        if self._proc is not None and self._proc.poll() is None:
            return True  # still alive

        app_id = self._find_running_app_id()
        if not app_id:
            return False

        self._app_id = app_id
        self._proc = subprocess.Popen(
            [MODAL_BIN, "app", "logs", app_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_modal_env(),
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        print(f"[{_timestamp()}] [SYNC] Log streamer connected to app {app_id}")
        sys.stdout.flush()
        return True

    def stop(self):
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def main():
    parser = argparse.ArgumentParser(description="Background Modal volume sync daemon")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"Sync interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--once", action="store_true",
                        help="Run a single sync pass and exit")
    parser.add_argument("--min-step", type=int, default=0,
                        help="Only forward log lines with global step >= this value (default: 0)")
    args = parser.parse_args()

    print(f"[{_timestamp()}] [SYNC] Volume sync daemon started.")
    print(f"[SYNC]   Volume:   {VOLUME_NAME}:/{VOLUME_PATH}")
    print(f"[SYNC]   Local:    {LOCAL_DEST}")
    print(f"[SYNC]   Interval: {args.interval}s")
    print(f"[SYNC]   Min step: {args.min_step}")
    print(f"[SYNC]   PID:      {os.getpid()}")
    sys.stdout.flush()

    if args.once:
        sync_once()
        return

    log_streamer = LogStreamer(min_global_step=args.min_step)

    while True:
        try:
            sync_once()
            log_streamer.ensure_running()
            if is_training_complete():
                print(f"[{_timestamp()}] [SYNC] Training appears complete (test_visualization or final checkpoint exists). Exiting.")
                log_streamer.stop()
                sys.stdout.flush()
                break
        except KeyboardInterrupt:
            print(f"\n[{_timestamp()}] [SYNC] Interrupted. Exiting.")
            log_streamer.stop()
            break
        except Exception as e:
            print(f"[{_timestamp()}] [SYNC] Loop error: {e}")
            sys.stdout.flush()

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
