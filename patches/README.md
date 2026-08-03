# Patches for the external Open-ECG-Digitizer checkout

These patches are applied **to the Open-ECG-Digitizer checkout**, never to files in this
repository. `scripts/setup_digitizer.sh` applies them for you.

## Licensing

Open-ECG-Digitizer is CC BY-SA 4.0. A patch against it is a modification of
ShareAlike-licensed material, so **the contents of this directory are licensed
CC BY-SA 4.0**, not under the repository's LICENSE. Keeping them here — isolated,
small, and never merged into `ecg_pipeline/` — is what keeps the rest of the
repository free of copyleft obligations.

If you share a patched digitizer, share it under CC BY-SA 4.0 and note the
modification, as the licence requires.

## `0001-digitizer-portability.patch`

Two environment fixes, neither of which changes digitization behaviour:

1. **`src/digitize.py`** — `decode_and_prepare_image` uses PIL instead of
   `torchvision.io.decode_image`. torchvision 0.17.2 (the last build available for
   Intel macOS) does not accept a file path or a string `mode` argument.
2. **`src/utils.py`** — the `ray.tune` import is made optional. `ray` is a training-only
   dependency; requiring it at import time breaks inference-only installs.

Both are generic bug fixes rather than anything specific to this project. **They belong
upstream** — sending them as a PR to `Ahus-AIM/Open-ECG-Digitizer` would let this
directory disappear entirely, which is the ideal end state.

Apply manually with:

    cd "$OPEN_ECG_DIGITIZER_HOME"
    git apply /path/to/patches/0001-digitizer-portability.patch
