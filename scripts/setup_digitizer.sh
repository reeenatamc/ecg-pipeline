#!/usr/bin/env bash
# Clone (if needed) and patch the external Open-ECG-Digitizer checkout.
#
# The digitizer is deliberately NOT vendored into this repository -- it is CC BY-SA 4.0
# and copying it would make this repo a derivative work. See NOTICE.
#
# Usage:
#   bash scripts/setup_digitizer.sh                 # sibling dir next to this repo
#   OPEN_ECG_DIGITIZER_HOME=/path bash scripts/setup_digitizer.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${OPEN_ECG_DIGITIZER_HOME:-$(dirname "$REPO_ROOT")/Open-ECG-Digitizer}"
UPSTREAM="https://github.com/Ahus-AIM/Open-ECG-Digitizer.git"
PATCH="$REPO_ROOT/patches/0001-digitizer-portability.patch"

if [ ! -f "$HOME_DIR/src/digitize.py" ]; then
  echo "==> Cloning Open-ECG-Digitizer into $HOME_DIR"
  echo "    (git-lfs is required for the model weights)"
  command -v git-lfs >/dev/null 2>&1 || { echo "error: git-lfs is not installed. brew install git-lfs"; exit 1; }
  git clone "$UPSTREAM" "$HOME_DIR"
  git -C "$HOME_DIR" lfs pull
else
  echo "==> Using existing checkout at $HOME_DIR"
fi

echo "==> Applying portability patch"
if git -C "$HOME_DIR" apply --check "$PATCH" 2>/dev/null; then
  git -C "$HOME_DIR" apply "$PATCH"
  echo "    applied"
elif git -C "$HOME_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "    already applied, skipping"
else
  echo "    WARNING: patch did not apply cleanly."
  echo "    Upstream may have changed. Review $PATCH by hand."
fi

echo
echo "Digitizer ready at: $HOME_DIR"
echo "Add this to your shell profile if it is not the default sibling location:"
echo "  export OPEN_ECG_DIGITIZER_HOME=\"$HOME_DIR\""
