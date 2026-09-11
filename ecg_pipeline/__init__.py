"""ECG pipeline: image -> digitized time series -> diagnostic interpretation.

Stage 1 (digitization) runs Open-ECG-Digitizer as an external program.
Stage 2 (interpretation) runs ECGFounder in-process.

See NOTICE for third-party attribution and licensing.
"""

__version__ = "0.1.1"
