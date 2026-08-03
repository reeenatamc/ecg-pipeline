# ecg-pipeline

Paper ECG image → digitized time series → diagnostic interpretation.

Two stages:

1. **Digitization** — [Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer)
   converts a scanned or photographed 12-lead ECG into canonical time series.
   It runs as an **external process**; its source is never bundled here.
2. **Interpretation** — [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)
   scores the digitized signal against 150 diagnostic classes.

## Why the digitizer is not vendored

Open-ECG-Digitizer is **CC BY-SA 4.0** — a ShareAlike (copyleft) licence. Copying its
source here would make this repository Adapted Material and force it under CC BY-SA 4.0
too. Running it as a separate program does not, and CC BY-SA has no network/SaaS clause,
so serving it from a backend triggers nothing. Its *output* is not encumbered either.

This is a structural property of the repo, not a detail. **Do not paste upstream source
into `ecg_pipeline/`.** Behaviour changes go in `configs/` or `patches/`. See
[NOTICE](NOTICE) for the full reasoning and the mandatory citation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/setup_digitizer.sh    # clone + patch the digitizer (uses an existing
                                   # checkout if you already have one)
bash scripts/download_weights.sh   # ECGFounder checkpoints, ~740 MB total
```

If your digitizer checkout lives somewhere non-standard:

```bash
export OPEN_ECG_DIGITIZER_HOME=/path/to/Open-ECG-Digitizer
```

The digitizer needs its own dependencies (`pip install -r $OPEN_ECG_DIGITIZER_HOME/requirements.txt`).
It can share this virtualenv or use its own — if separate, pass the interpreter through
`digitize(python_exe=...)`.

## Usage

```bash
# Full pipeline
python -m ecg_pipeline --images path/to/images --out path/to/output

# Right-sided limb leads
python -m ecg_pipeline --images in/ --out out/ --lead-layout lead_layouts_limbaug_right.yml

# Digitization only
python -m ecg_pipeline --images in/ --out out/ --digitize-only

# Machine-readable
python -m ecg_pipeline --images in/ --out out/ --json
```

Per ECG the pipeline writes `<name>_timeseries_canonical.csv` and
`<name>_interpretation.json` into the output directory.

As a library:

```python
from ecg_pipeline import pipeline

results = pipeline.run(image_dir="in/", output_dir="out/", pathway="rhythm")
```

## Interpretation pathways

| Pathway | What it does | Trust |
|---|---|---|
| `rhythm` (default) | 1-lead model over every full-length rhythm strip (II/V1/V5), opinions averaged | Reliable for rhythm and rate |
| `1lead` | 1-lead model on one chosen lead | Inspection/debugging |
| `12lead` | Per-lead ~2.5 s windows assembled into a montage → 12-lead model | **Experimental** |

A standard 3×4 paper ECG prints only ~2.5 s of most leads, and the columns are recorded
at *different times*. The `12lead` pathway therefore feeds the model phase-misaligned
beats and over-calls pathology (e.g. a false `LATERAL INFARCT` on a normal ECG). It needs
representative-beat / R-peak alignment before it can be trusted.

**Scores are rankings, not calibrated probabilities.** Raw sigmoid outputs are not
present/absent decisions — `ABNORMAL ECG` can score high alongside `NORMAL ECG`.
ECGFounder binarizes with per-class thresholds derived on PTB-XL. Pass `--thresholds`
(a `{label: threshold}` JSON) or `--threshold` to get a flagged list; without one you get
a ranking only.

## Quality gating

A digitizer that fails still produces a CSV, and ECGFounder will happily score noise at
0.99. Every result therefore carries `degraded`, `warnings`, `digitization`, and
`signal_quality`. **Read `degraded` before `topk`.**

Two independent gates, both needed — neither catches the other's failures:

1. **Layout** (from the digitizer's `digitization_metadata.csv`). When no layout matches,
   the digitizer emits `lead_layout: "Unknown layout"`, canonicalization returns an
   all-NaN frame, and whatever signal survives was recovered by rhythm-strip cosine
   matching — so *which trace is which lead* is a guess. Always flagged.
2. **Coverage** (from the signal itself). The rhythm pathway is only meaningful when a
   lead was printed at full length. If none reaches `--coverage-min`, the result is built
   from a single ~2.5 s fragment; that is flagged rather than passed off as an ensemble.

Observed on the sample ECGs:

| Input | Layout | Leads | Verdict |
|---|---|---|---|
| clean scan | `standard_3x4_with_r3` | 12/12, rhythm II/V1/V5 | clean — `SINUS RHYTHM 0.99` |
| low-res scan | `standard_3x4_with_r1` | none ≥60% | `DEGRADED` — fell back to V2 alone |
| unmatched layout | `Unknown layout` | 3/12 | `DEGRADED` — lead identity unreliable |

The second case passes the layout gate and the third passes the coverage gate, which is
why both exist.

In a backend, use `--fail-on-degraded` (exit code 2):

```bash
python -m ecg_pipeline --images in/ --out out/ --fail-on-degraded
```

`matching_cost` is reported but **not thresholded by default**. It is an unbounded
residual (mean grid distance × a scaling factor), not a normalized score, and `1.0` is a
hardcoded sentinel for "no layout matched" rather than a measurement. Calibrate on your
own data before enabling `--max-matching-cost`.

## Tests

```bash
python -m unittest discover -s tests
```

37 tests, ~0.04 s. They need neither the model weights nor a digitizer checkout — the
safety logic runs on synthetic arrays, and `digitize()` is exercised against a mocked
subprocess. Written with stdlib `unittest` so a fresh clone can run them before any
`pip install`.

The suite covers the quality gates, the metadata parsing, checkout resolution, and the
argument construction that the digitizer's CLI is picky about. Two cases worth knowing:
`test_zscore_is_unit_invariant` is the regression guard for the µV/mV mismatch (see
above), and `test_input_and_output_are_passed_as_positional_overrides` pins the fact
that the digitizer takes overrides positionally — a `--overrides` flag makes it exit 2.

## Layout

```
ecg_pipeline/
  digitizer.py          external-process wrapper (licensing boundary lives here)
  pipeline.py           orchestration
  cli.py                command line
  interpret/            ECGFounder interpretation (net1d.py vendored, MIT)
configs/                digitizer configs and lead layouts
patches/                fixes applied to the external checkout (CC BY-SA 4.0)
scripts/                setup helpers
tests/                  stdlib unittest suite, no weights required
```

## Status

Not a medical device. No regulatory clearance. Research and development use only —
outputs are not clinical decisions.

Licensed proprietary; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
