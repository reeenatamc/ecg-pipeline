# Integration fixtures

`images/normal.png` (4.2 MB) is copied from Open-ECG-Digitizer's own sandbox
(`sandbox/in_normal/normal.png` in that checkout), where it is the reference clean scan
the digitizer's own maintainers test against. Committing a 4 MB PNG here is deliberate:
this is the one place the pipeline is checked against the real digitizer and real model
weights end to end, and that needs a real scanned image, not a synthetic array.

`expected/normal_timeseries_canonical.csv` is the canonical CSV `scripts/integration_check.py`
compares fresh runs against, plus `expected/normal_expected.json` recording the layout
name from the same run. Regenerate both with:

```bash
python scripts/integration_check.py --update-reference
```

only when a change to `patches/`, `configs/`, or the digitizer checkout itself is expected
to change the digitized output, and only after checking the new output by hand -- this
file is the ground truth every future run is judged against.

See the root [README.md](../../README.md#tests) for when to run the check itself.

## A known source of flakiness: V5's coverage is bimodal

On this image, `standard_3x4_with_r3`'s wildcard rhythm strip is attributed to V5, and its
recovered length is not perfectly reproducible: repeated runs of the identical image
through the identical checkout land on one of two coverage values, about 0.85 or about
0.996, roughly at random. Confirmed not to be caused by CPU thread scheduling
(`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` still shows both values
across repeated runs) or by Python's hash randomization (`PYTHONHASHSEED=0` likewise still
shows both) -- so it is inherent to the digitizer's own CPU inference (most likely
non-deterministic floating-point reduction inside a torch op), not anything under this
repo's control. Every other field -- the layout, the full set of leads with signal, every
other lead's coverage, and V5's own correlation on the overlap (still 1.0000 either way,
so the extra/missing stretch is real recovered trace, not noise) -- has been observed
stable across more than a dozen runs.

If `integration_check.py` fails on V5's coverage alone, with everything else matching and
V5's correlation still ≥ 0.99, re-run once before treating it as a real regression -- it is
very likely this known bimodality, not a change caused by whatever you touched. A failure
on any other lead, on the layout, or on the set of leads with signal is not this and should
be treated as a real finding.
