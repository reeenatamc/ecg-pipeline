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
```

## Status

Not a medical device. No regulatory clearance. Research and development use only —
outputs are not clinical decisions.

Licensed proprietary; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
