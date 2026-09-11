"""Tests for what the pipeline does with a whole batch, rather than with one good ECG.

The failure these cover is the same one the quality gates cover for signals: a run that
went wrong looking exactly like a run that went right. An image the digitizer skipped, or a
--digitize-only run whose layout was never identified, both used to come back clean.

``digitizer.digitize`` is mocked -- these exercise orchestration, not digitization.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from ecg_pipeline import cli, pipeline
from ecg_pipeline.digitizer import CANONICAL_SUFFIX
from ecg_pipeline.interpret.waveform import CANONICAL_LEADS

N = 500

# A small stand-in for the digitizer's own lead-layout templates, covering every layout
# name this file uses. Patched into every ``PipelineRunCase`` test (see ``setUp`` below) so
# the gate-4/5 logic in ``pipeline.py`` never depends on a real Open-ECG-Digitizer checkout
# being present -- these tests run on synthetic arrays only, per the README's promise.
# Real-template parsing (nested grids, "-" markers, "Any" wildcards) is covered separately
# in tests/test_layout_templates.py against an actual excerpt of upstream's file.
TEST_LAYOUTS = {
    "cabrera_6x1_limb": {"leads": ["aVL", "I", "-aVR", "II", "aVF", "III"], "rhythm_leads": []},
    "standard_3x1": {"leads": ["I", "II", "III"], "rhythm_leads": []},
    "standard_3x4_with_r1": {
        "leads": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
        "rhythm_leads": ["Any"],
    },
    "standard_3x4_with_r3": {
        "leads": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
        "rhythm_leads": ["Any", "Any", "Any"],
    },
    "standard_12x1": {"leads": list(CANONICAL_LEADS), "rhythm_leads": []},
}


def write_canonical_csv(path: Path, coverage: dict[str, float]) -> None:
    """A canonical CSV where each lead is valid for the given fraction of the record."""
    data = np.full((N, len(CANONICAL_LEADS)), np.nan)
    for column, lead in enumerate(CANONICAL_LEADS):
        valid = int(round(coverage.get(lead, 0.0) * N))
        if valid:
            data[:valid, column] = np.linspace(-1.0, 1.0, valid)
    with path.open("w") as fh:
        fh.write(",".join(CANONICAL_LEADS) + "\n")
        np.savetxt(fh, data, delimiter=",")


def write_metadata(output_dir: Path, rows: dict[str, str]) -> None:
    lines = ["file_path,matching_cost,is_flipped,lead_layout"]
    lines += [f"{name},0.4,False,{layout}" for name, layout in rows.items()]
    (output_dir / "digitization_metadata.csv").write_text("\n".join(lines) + "\n")


class PipelineRunCase(unittest.TestCase):
    """Runs ``pipeline.run`` against a fake digitizer that writes whatever we tell it to."""

    def setUp(self) -> None:
        self.images = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(pipeline, "_layout_definitions", return_value=TEST_LAYOUTS))

    def add_image(self, name: str) -> None:
        # Wide enough that 'auto' leaves it alone; one pixel row tall so that encoding it is
        # free. These tests are about orchestration, not pixels.
        Image.new("RGB", (2400, 4), "white").save(self.images / f"{name}.png")

    def run_pipeline(self, produces: dict[str, dict[str, float]], layouts: dict[str, str], **kwargs):
        """``produces`` maps record name -> per-lead coverage of the CSV the digitizer writes."""

        def fake_digitize(*_args, **_kwargs):
            paths = []
            for name, coverage in produces.items():
                path = self.output / f"{name}{CANONICAL_SUFFIX}"
                write_canonical_csv(path, coverage)
                paths.append(path)
            write_metadata(self.output, layouts)
            return sorted(paths)

        with mock.patch.object(pipeline.digitizer, "digitize", side_effect=fake_digitize):
            return pipeline.run(
                image_dir=self.images,
                output_dir=self.output,
                skip_interpretation=True,
                quiet=True,
                **kwargs,
            )


class TestDigitizeOnlyGates(PipelineRunCase):
    def test_an_unidentified_layout_is_flagged_without_interpretation(self):
        # Previously --digitize-only returned no 'degraded' key at all, so --fail-on-degraded
        # had nothing to fail on however badly digitization had gone.
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "Unknown layout"})

        self.assertTrue(results[0]["degraded"])
        self.assertTrue(any("could not identify the lead layout" in w for w in results[0]["warnings"]))

    def test_a_clean_digitization_is_not_flagged(self):
        self.add_image("ecg")
        # standard_3x4_with_r3's realistic shape: every grid lead has some signal (its ~2.5s
        # column), and the three wildcard rhythm leads landed on the conventional strips.
        coverage = {lead: 0.25 for lead in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0}

        results = self.run_pipeline({"ecg": coverage}, {"ecg": "standard_3x4_with_r3"})

        self.assertFalse(results[0]["degraded"])
        self.assertEqual(results[0]["warnings"], [])

    def test_no_full_length_lead_is_flagged_without_interpretation(self):
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {lead: 0.25 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"})

        self.assertTrue(results[0]["degraded"])
        self.assertIn("signal_quality", results[0])

    def test_the_digitize_only_path_does_not_pull_in_torch(self):
        # The whole reason assess_quality and load_canonical_csv live in waveform.py. Checked
        # in a fresh interpreter because by this point in the suite another module has
        # already imported torch, and sys.modules would say so no matter what this path does.
        probe = (
            "import sys; from ecg_pipeline import pipeline; "
            "from ecg_pipeline.interpret.waveform import assess_quality, load_canonical_csv; "
            "print('torch' in sys.modules)"
        )
        finished = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )

        self.assertEqual(finished.returncode, 0, finished.stderr)
        self.assertEqual(finished.stdout.strip(), "False")


class TestLayoutTemplateGates(PipelineRunCase):
    """Gate 4 (fewer leads than the matched template defines) and gate 5 (a wildcard
    rhythm strip landing on an unconventional lead). Both need the layout name together
    with the signal, which coverage-only gate 3 in ``assess_quality`` never sees.

    ``pipeline._layout_definitions`` is mocked (with ``TEST_LAYOUTS``, in ``setUp`` above)
    rather than pointed at a real checkout: the parsing itself (nested grids, "-" markers,
    "Any" wildcards) is covered in tests/test_layout_templates.py against a real excerpt,
    so these only need to know the gates react correctly to a given template shape.
    """

    def test_fewer_leads_than_the_template_defines_degrades(self):
        # case_10: cabrera_6x1_limb defines 6 leads, only 5 came back with signal.
        self.add_image("case_10")
        coverage = {l: 1.0 for l in ("I", "II", "III", "aVR", "aVL")}  # aVF missing

        results = self.run_pipeline({"case_10": coverage}, {"case_10": "cabrera_6x1_limb"})

        self.assertTrue(results[0]["degraded"])
        self.assertTrue(any("defines 6 lead" in w and "missing aVF" in w for w in results[0]["warnings"]))
        self.assertEqual(results[0]["gates"], ["leads-missing-from-template"])

    def test_the_full_template_does_not_degrade(self):
        # standard_3x1 defines only 3 leads and all 3 came back with signal; the pipeline
        # still warns "3 of 12" (gate 3, unrelated to this one) but must not degrade for it,
        # and gate 4 itself must add nothing since nothing is missing from the template.
        self.add_image("infarct")

        results = self.run_pipeline({"infarct": {l: 1.0 for l in ("I", "II", "III")}}, {"infarct": "standard_3x1"})

        self.assertFalse(results[0]["degraded"])
        self.assertFalse(any("defines 3 lead" in w for w in results[0]["warnings"]))

    def test_a_full_twelve_lead_match_stays_clean(self):
        # "12 of 12" means every lead carries some signal, not that every lead is full
        # length -- on a real 3x4 print only the wildcard strips (here landing on the
        # conventional II/V1/V5) ever reach full length; the other 9 are ~2.5s columns.
        self.add_image("normal")
        coverage = {l: 0.25 for l in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0}

        results = self.run_pipeline({"normal": coverage}, {"normal": "standard_3x4_with_r3"})

        self.assertFalse(results[0]["degraded"])
        self.assertEqual(results[0]["warnings"], [])

    def test_an_unavailable_layout_definition_warns_without_degrading(self):
        self.add_image("ecg")
        coverage = {l: 0.25 for l in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0}

        with mock.patch.object(pipeline, "_layout_definitions", return_value=None):
            results = self.run_pipeline({"ecg": coverage}, {"ecg": "standard_3x4_with_r3"})

        self.assertFalse(results[0]["degraded"])
        self.assertTrue(any("was unavailable" in w for w in results[0]["warnings"]))

    def test_an_unconventional_rhythm_strip_degrades(self):
        # afib_E000735-like: the wildcard strip on standard_3x4_with_r1 was identified as
        # aVF instead of the printed II.
        self.add_image("afib_E000735")
        coverage = {l: 0.25 for l in CANONICAL_LEADS} | {"aVF": 1.0}

        results = self.run_pipeline({"afib_E000735": coverage}, {"afib_E000735": "standard_3x4_with_r1"})

        self.assertTrue(results[0]["degraded"])
        self.assertTrue(any("unverified" in w and "aVF" in w for w in results[0]["warnings"]))
        self.assertEqual(results[0]["gates"], ["rhythm-strip-unverified"])

    def test_a_conventional_rhythm_strip_does_not_degrade(self):
        self.add_image("afib_E000742")
        coverage = {l: 0.25 for l in CANONICAL_LEADS} | {"II": 1.0}

        results = self.run_pipeline({"afib_E000742": coverage}, {"afib_E000742": "standard_3x4_with_r1"})

        self.assertFalse(results[0]["degraded"])
        self.assertEqual(results[0]["warnings"], [])

    def test_a_12x1_layout_never_triggers_the_rhythm_strip_check(self):
        # Every lead is full length and there is no wildcard rhythm lead at all, so gate 5
        # must not fire just because most of them are not II/V1/V5.
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {l: 1.0 for l in CANONICAL_LEADS}}, {"ecg": "standard_12x1"})

        self.assertFalse(results[0]["degraded"])
        self.assertEqual(results[0]["warnings"], [])


class TestGateIds(PipelineRunCase):
    """``result["gates"]``: the stable ids ``contract.failure_reason`` reads instead of
    parsing warning text. Covers the digitize-only branch; the interpretation branch's own
    wiring (it assembles the same list from ``result["signal_quality"]["gates"]`` plus
    ``"error" in result``) is covered in ``TestInterpretationBranchGateIds`` below.
    """

    def test_layout_unknown_reports_its_gate_id(self):
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "Unknown layout"})

        self.assertEqual(results[0]["gates"], ["layout-unknown"])

    def test_no_signal_reports_its_gate_id(self):
        # With truly nothing recovered, gate 4 also fires (the template's own leads are all
        # missing too) -- both ids are reported, in the order their gates run.
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {}}, {"ecg": "standard_3x1"})

        self.assertEqual(results[0]["gates"], ["no-signal", "leads-missing-from-template"])

    def test_no_full_length_lead_reports_its_gate_id(self):
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {lead: 0.25 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"})

        self.assertEqual(results[0]["gates"], ["no-full-length-lead"])

    def test_a_clean_record_reports_no_gates(self):
        self.add_image("ecg")
        coverage = {lead: 0.25 for lead in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0}

        results = self.run_pipeline({"ecg": coverage}, {"ecg": "standard_3x4_with_r3"})

        self.assertEqual(results[0]["gates"], [])

    def test_layout_unknown_and_no_signal_can_both_fire(self):
        # An unmatched layout also has no leads_with_signal on this fixture: both gates
        # apply and both ids are reported, layout-unknown first.
        self.add_image("ecg")

        results = self.run_pipeline({"ecg": {}}, {"ecg": "Unknown layout"})

        self.assertEqual(results[0]["gates"], ["layout-unknown", "no-signal"])


class TestInterpretationBranchGateIds(PipelineRunCase):
    """Same ``gates`` field, assembled in ``pipeline.run``'s interpretation branch instead
    of the digitize-only one. ``interpret_ecg.interpret_csv`` is mocked, as in
    ``TestModelIsLoadedOnce``, so these test the wiring rather than a real interpretation.
    """

    def run_with_fake_interpret(self, produces, layouts, fake_interpret):
        self.add_image(next(iter(produces)))

        def fake_digitize(*_args, **_kwargs):
            paths = []
            for name, coverage in produces.items():
                path = self.output / f"{name}{CANONICAL_SUFFIX}"
                write_canonical_csv(path, coverage)
                paths.append(path)
            write_metadata(self.output, layouts)
            return sorted(paths)

        import ecg_pipeline.interpret.interpret_ecg as interpret

        with (
            mock.patch.object(pipeline.digitizer, "digitize", side_effect=fake_digitize),
            mock.patch.object(interpret, "build_model_for", return_value=(object(), "ckpt.pth")),
            mock.patch.object(interpret, "interpret_csv", side_effect=fake_interpret),
        ):
            return pipeline.run(image_dir=self.images, output_dir=self.output, quiet=True)

    def test_a_gate_2_id_from_signal_quality_passes_through(self):
        def fake_interpret(csv_path, **_kwargs):
            return {
                "source_csv": csv_path,
                "topk": [],
                "summary": {},
                "degraded": True,
                "warnings": [],
                "signal_quality": {"gates": ["no-full-length-lead"]},
            }

        results = self.run_with_fake_interpret(
            {"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"}, fake_interpret
        )

        self.assertEqual(results[0]["gates"], ["no-full-length-lead"])

    def test_an_interpretation_exception_reports_its_gate_id(self):
        def fake_interpret(csv_path, **_kwargs):
            raise RuntimeError("boom")

        results = self.run_with_fake_interpret(
            {"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"}, fake_interpret
        )

        self.assertEqual(results[0]["gates"], ["interpretation-error"])

    def test_a_clean_interpretation_reports_no_gates(self):
        def fake_interpret(csv_path, **_kwargs):
            return {
                "source_csv": csv_path,
                "topk": [],
                "summary": {},
                "degraded": False,
                "warnings": [],
                "signal_quality": {"gates": []},
            }

        results = self.run_with_fake_interpret(
            {"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"}, fake_interpret
        )

        self.assertEqual(results[0]["gates"], [])


class TestMissingRecords(PipelineRunCase):
    def test_an_image_the_digitizer_skipped_is_reported_as_failed(self):
        # The digitizer catches its own per-image errors and still exits 0, so a skipped
        # image simply produces no CSV. A batch of two that lost one used to come back as a
        # clean run of one.
        self.add_image("good")
        self.add_image("bad")

        results = self.run_pipeline({"good": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"good": "standard_3x4_with_r3"})

        self.assertEqual(len(results), 2)
        failed = [r for r in results if r["record"] == "bad"][0]
        self.assertTrue(failed["degraded"])
        self.assertIn("no output", failed["error"])
        self.assertIsNone(failed.get("source_csv"))

    def test_a_failed_image_still_reports_what_was_sent_to_the_digitizer(self):
        self.add_image("bad")

        results = self.run_pipeline({}, {})

        self.assertEqual(results[0]["preprocessing"]["original_size"], [2400, 4])

    def test_nothing_is_invented_when_every_image_succeeds(self):
        self.add_image("a")
        self.add_image("b")

        results = self.run_pipeline(
            {name: {lead: 1.0 for lead in CANONICAL_LEADS} for name in ("a", "b")},
            {"a": "standard_3x4_with_r3", "b": "standard_3x4_with_r3"},
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all("error" not in r for r in results))


class TestNestedBatches(PipelineRunCase):
    def add_nested_image(self, subdirectory: str, name: str) -> None:
        (self.images / subdirectory).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2400, 4), "white").save(self.images / subdirectory / f"{name}.png")

    def test_metadata_is_read_from_every_subdirectory(self):
        # The digitizer writes one digitization_metadata.csv per output subdirectory, each
        # row naming a bare basename. Reading only the root one left every nested record
        # with "digitization quality is unverified".
        self.add_nested_image("batch-1", "ecg")
        (self.output / "batch-1").mkdir(parents=True)
        write_canonical_csv(self.output / "batch-1" / f"ecg{CANONICAL_SUFFIX}", {"II": 1.0})
        write_metadata(self.output / "batch-1", {"ecg": "Unknown layout"})

        def fake_digitize(*_args, **_kwargs):
            return [self.output / "batch-1" / f"ecg{CANONICAL_SUFFIX}"]

        with mock.patch.object(pipeline.digitizer, "digitize", side_effect=fake_digitize):
            results = pipeline.run(image_dir=self.images, output_dir=self.output, skip_interpretation=True, quiet=True)

        self.assertEqual(results[0]["record"], "batch-1/ecg")
        self.assertEqual(results[0]["digitization"]["lead_layout"], "Unknown layout")
        self.assertTrue(results[0]["degraded"])


class TestStaleOutputs(PipelineRunCase):
    def test_a_stale_csv_is_not_read_as_a_fresh_result(self):
        # The digitizer no longer empties the output directory, so a record that succeeded
        # last run and fails this one must not have its old CSV picked up as this run's.
        self.add_image("ecg")
        stale = self.output / f"ecg{CANONICAL_SUFFIX}"
        write_canonical_csv(stale, {lead: 1.0 for lead in CANONICAL_LEADS})

        results = self.run_pipeline({}, {})  # the digitizer writes nothing this time

        self.assertFalse(stale.exists())
        self.assertIn("no output", results[0]["error"])

    def test_output_from_an_unrelated_run_is_left_alone(self):
        self.add_image("ecg")
        other = self.output / f"someone-elses{CANONICAL_SUFFIX}"
        write_canonical_csv(other, {"II": 1.0})
        notes = self.output / "notes.txt"
        notes.write_text("not ours")

        results = self.run_pipeline({"ecg": {lead: 1.0 for lead in CANONICAL_LEADS}}, {"ecg": "standard_3x4_with_r3"})

        # Neither deleted nor mistaken for a result of this run.
        self.assertTrue(other.exists())
        self.assertTrue(notes.exists())
        self.assertEqual([r["record"] for r in results], ["ecg"])


class TestModelIsLoadedOnce(PipelineRunCase):
    def test_the_checkpoint_is_read_once_for_the_whole_batch(self):
        # It used to be built inside interpret_csv, so a batch of 50 paid 50 reads of a
        # 370 MB checkpoint. interpret_csv still builds its own when given none, which is
        # what the standalone entry point needs.
        for name in ("a", "b", "c"):
            self.add_image(name)
        produces = {name: {lead: 1.0 for lead in CANONICAL_LEADS} for name in ("a", "b", "c")}
        layouts = dict.fromkeys(produces, "standard_3x4_with_r3")

        def fake_digitize(*_args, **_kwargs):
            paths = []
            for name, coverage in produces.items():
                path = self.output / f"{name}{CANONICAL_SUFFIX}"
                write_canonical_csv(path, coverage)
                paths.append(path)
            write_metadata(self.output, layouts)
            return sorted(paths)

        sentinel = object()
        interpreted: list[object] = []

        def fake_interpret(csv_path, **kwargs):
            interpreted.append(kwargs["model"])
            return {"source_csv": csv_path, "topk": [], "summary": {}, "degraded": False, "warnings": []}

        import ecg_pipeline.interpret.interpret_ecg as interpret

        with (
            mock.patch.object(pipeline.digitizer, "digitize", side_effect=fake_digitize),
            mock.patch.object(interpret, "build_model_for", return_value=(sentinel, "ckpt.pth")) as build,
            mock.patch.object(interpret, "interpret_csv", side_effect=fake_interpret),
        ):
            pipeline.run(image_dir=self.images, output_dir=self.output, quiet=True)

        self.assertEqual(build.call_count, 1)
        self.assertEqual(interpreted, [sentinel] * 3)

    def test_no_checkpoint_is_read_when_the_digitizer_produced_nothing(self):
        self.add_image("bad")
        import ecg_pipeline.interpret.interpret_ecg as interpret

        with (
            mock.patch.object(pipeline.digitizer, "digitize", return_value=[]),
            mock.patch.object(interpret, "build_model_for") as build,
        ):
            results = pipeline.run(image_dir=self.images, output_dir=self.output, quiet=True)

        build.assert_not_called()
        self.assertEqual(results[0]["record"], "bad")


class TestCliExitCodes(unittest.TestCase):
    def run_main(self, results, *extra):
        # stderr redirected only to keep the suite's own output readable.
        with mock.patch.object(cli.pipeline, "run", return_value=results), contextlib.redirect_stderr(io.StringIO()):
            return cli.main(["--images", "in", "--out", "out", "--quiet", *extra])

    def test_a_batch_where_every_image_failed_exits_1(self):
        # Not 0: the run produced a result list, but not one digitized ECG.
        self.assertEqual(self.run_main([{"record": "a", "error": "boom", "degraded": True}]), 1)

    def test_a_partial_batch_still_succeeds(self):
        results = [
            {"record": "a", "source_csv": "a.csv", "degraded": False},
            {"record": "b", "error": "boom", "degraded": True},
        ]

        self.assertEqual(self.run_main(results), 0)

    def test_fail_on_degraded_catches_a_skipped_image(self):
        results = [
            {"record": "a", "source_csv": "a.csv", "degraded": False},
            {"record": "b", "error": "boom", "degraded": True},
        ]

        self.assertEqual(self.run_main(results, "--fail-on-degraded"), 2)


if __name__ == "__main__":
    unittest.main()
