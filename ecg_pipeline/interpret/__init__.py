"""ECG interpretation with ECGFounder (PKUDigitalHealth, MIT licence).

``net1d.py`` is vendored verbatim from the ECGFounder repository under MIT; see NOTICE.
``interpret_ecg.py`` is original work in this repository.
"""

# Declared here rather than in ``interpret_ecg`` so that the CLI can offer the choices
# without importing torch and a 370 MB checkpoint just to build its argument parser.
PATHWAYS = ("rhythm", "1lead", "morphology", "12lead")
