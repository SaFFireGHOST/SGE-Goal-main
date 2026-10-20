#!/usr/bin/env bash
#
# Stop the LOCAL processes for the i1 ablation (the detached `modal run` client and
# the sync poller). NOTE: because training runs DETACHED on Modal, killing the local
# client does NOT stop the remote training run. To stop the remote run, use:
#   python -m modal app stop flux2-klein-calvin-bft-i1
#
cd "$(dirname "$0")"
for f in train.pid sync.pid; do
  if [ -f "$f" ]; then
    pid="$(cat "$f")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "stopped $f (PID $pid)"
    else
      echo "$f (PID $pid) not running"
    fi
    rm -f "$f"
  fi
done
