# Patches for the external Open-ECG-Digitizer checkout

These patches are applied **to the Open-ECG-Digitizer checkout**, never to files in this
repository. `scripts/setup_digitizer.sh` applies them for you, in filename order --
each one is a diff against the tree the previous ones left behind.

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

## `0002-rhythm-strip-label.patch`

Fixes a real digitization bug found in audit: on a standard 3×4 print with a wildcard
rhythm strip (`rhythm_leads: ["Any"]` in `src/config/lead_layouts_all.yml`), the
digitizer had no way to know which lead the strip actually was. `LeadIdentifier`
assigned it to a canonical lead purely by cosine similarity against the grid traces —
but limb leads are morphologically similar (II, III and aVF in particular), so on real
images this misassigned strips clearly printed "II" to aVF or V3.

The strip's lead name is usually printed right under it, and the lead-name U-Net
already segments that text for the whole image — `LeadIdentifier` just never looked at
it for wildcard rows. **`src/model/lead_identifier.py`** now checks, for each wildcard
rhythm row, whether the U-Net's highest-confidence label *within that row's own pixel
band* names an unclaimed canonical lead with confidence ≥ 0.9 (`RHYTHM_LABEL_MIN_CONF`);
if so it uses that label, otherwise it falls back to the original cosine-similarity
assignment. The row band is derived from the neighbouring rows' trace positions
(midpoints), not the row's own signal excursion, because a quiet rhythm strip can swing
far less than the space it occupies on the page. The confidence check matters: the same
digit-like glyphs (I/II/III) are prone to bleeding into each other's channel, so a
low-confidence label is worse than no label — the 0.9 bar was chosen empirically against
the audit's real image set, where genuine matches score ≥ 0.97 and ambiguous ones stay
well under 0.8.

This is a bug in upstream's own cosine-similarity logic, not something specific to how
this project drives the digitizer. **It belongs upstream** — sending it as a PR to
`Ahus-AIM/Open-ECG-Digitizer` would let this file disappear from the patch set.

Apply manually with:

    cd "$OPEN_ECG_DIGITIZER_HOME"
    git apply /path/to/patches/0002-rhythm-strip-label.patch

## `0003-digitizer-determinism.patch`

**This one does change digitization behaviour**, like `0002`. It is here because
without it the digitizer does not give the same answer twice.

Inference draws at random in four places, none of them seeded:

| file | what it draws |
|---|---|
| `src/model/perspective_detector.py:42` | subsample of non-zero pixels for the Hough transform |
| `src/model/perspective_detector.py:251` | subsample used to estimate a quantile |
| `src/model/dewarper.py:440` | subsample of warp control points |
| `src/model/signal_extractor.py:173` | jitter added to break ties |

The first is the one that bites: the sheet's perspective is estimated from a random
sample of pixels, so a different sample straightens the sheet slightly differently, the
grid lands slightly differently, and the identified layout can come out different.

Measured on one 1800x1076 photograph of a printout, eight identical runs at `--upscale 2`:

| runs | `matching_cost` | `lead_layout` |
|---:|---|---|
| 4 | 0.4591966730372528 | `precordial_6x1` |
| 2 | 0.44132470802308826 | `precordial_6x1` |
| 2 | 0.4550414180108963 | `standard_3x4_with_r3` |

The layout decides which trace is named which lead, so those are two different readings
of one image. Note also that `matching_cost` does not separate them: the wrong layout
scores better.

The patch seeds `random`, `numpy` and `torch` at the top of `process_one_file`, from a
hash of the decoded pixels. Keyed on the image rather than seeded once per process, so a
file digitizes the same way regardless of what else sits in the input directory and in
what order -- a single startup seed would leave each image depending on how many draws
the images before it consumed.

**It makes the result repeatable, not correct.** The variance is still there; the patch
fixes which draw you get. The variance also grows with image size, because the subsample
caps are constants while the pixel count is not -- which is why the same image varies
more at `--upscale 2` than at `off`. Raising those caps, or scaling them with the image,
would be the fix for the variance itself, and belongs upstream.

**This one does not belong upstream as-is.** Seeding from the image is a choice about
which of several answers you keep, and upstream should decide that for itself. What does
belong upstream is the report: unseeded sampling makes the digitizer irreproducible.
