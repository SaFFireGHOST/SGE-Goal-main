#!/bin/bash
# Install Octo at the commit we used, with our two patches applied:
#   octo/data/dataset.py              OCTO_DESYNC_TASK_AUG=1 gives the goal image its own
#                                     augmentation seed (see README)
#   octo/utils/visualization_lib.py   matplotlib >= 3.10 compatibility
#
# usage: bash setup_octo.sh [target_dir]      (default: ./octo)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HERE/octo}"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/octo-models/octo.git "$DEST"
fi
cd "$DEST"
git checkout 241fb35
git apply --check "$HERE/octo_patch/octo-241fb35-calvin.patch" 2>/dev/null \
  && git apply "$HERE/octo_patch/octo-241fb35-calvin.patch" \
  || echo "patch already applied (or local changes present); leaving the tree as is"
pip install -e .
echo "Octo installed at $DEST"
