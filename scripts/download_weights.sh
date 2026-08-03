#!/usr/bin/env bash
# Download the ECGFounder checkpoints (MIT licence) into weights/.
#
# ~370 MB each. They are gitignored -- large binaries do not belong in the repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ECGFOUNDER_WEIGHTS_DIR:-$REPO_ROOT/weights}"
BASE="https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main"

mkdir -p "$DEST"

for f in 1_lead_ECGFounder.pth 12_lead_ECGFounder.pth; do
  if [ -f "$DEST/$f" ]; then
    echo "==> $f already present, skipping"
    continue
  fi
  echo "==> Downloading $f (~370 MB)"
  curl -fL --progress-bar -o "$DEST/$f" "$BASE/$f"
done

echo
echo "Weights ready in: $DEST"
