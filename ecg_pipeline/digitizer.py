"""Run Open-ECG-Digitizer as an external program.

Licensing boundary — read before changing this file
---------------------------------------------------
Open-ECG-Digitizer is licensed CC BY-SA 4.0, a ShareAlike (copyleft) licence. This
module *invokes* it as a separate process and reads the files it writes. No upstream
source is copied, vendored, imported, or modified anywhere in this repository, so
nothing here is "Adapted Material" and the ShareAlike term is never triggered.

Keep it that way. If you ever need to change digitizer behaviour, add an override in
``configs/`` or a patch under ``patches/`` (applied to the external checkout) rather
than pasting upstream code into this package. See NOTICE for the full attribution.

Two ways to run it
------------------
``persistent`` (the default) starts the digitizer's ``src.serve`` entry point (added by
``patches/0004``) once, keeps it alive with its models loaded, and sends it one request
per call over stdin/stdout. It is still a separate process, so the boundary above holds.
It saves the imports and weight loading, about five seconds per study on a laptop CPU.

``subprocess`` launches ``python -m src.digitize`` for every call, as this module always
did. Select it with ``ECG_DIGITIZER_MODE=subprocess`` or ``digitize(..., mode=...)``, for
instance when the resident memory of a long-lived digitizer is not affordable.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import IO, Any

from ecg_pipeline import threads

ENV_HOME = "OPEN_ECG_DIGITIZER_HOME"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "digitizer_cpu.yml"

# Filename suffix the digitizer appends to each input image's basename.
CANONICAL_SUFFIX = "_timeseries_canonical.csv"

ENV_MODE = "ECG_DIGITIZER_MODE"
MODE_PERSISTENT = "persistent"
MODE_SUBPROCESS = "subprocess"
MODES = (MODE_PERSISTENT, MODE_SUBPROCESS)

# Long side, in pixels, the segmentation works at. The default lives in
# configs/digitizer_cpu.yml (MODEL.KWARGS.resample_size); this variable overrides it per
# deployment. See docs/RENDIMIENTO.md for how the default was chosen.
ENV_RESAMPLE_SIZE = "ECG_DIGITIZER_RESAMPLE_SIZE"
RESAMPLE_KEY = "MODEL.KWARGS.resample_size"
# The digitizer upsamples anything whose short side is under 512 px, so a smaller working
# size would contradict itself.
MIN_RESAMPLE_SIZE = 512


LAYOUT_KEY = "MODEL.KWARGS.config.LAYOUT_IDENTIFIER.config_path"


def lead_layout_override(layout: str | Path) -> dict[str, str]:
    """Build the override that swaps in a lead-layout file from this repo's ``configs/``.

    The digitizer opens ``config_path`` relative to its own root, so a layout living here
    has to be passed as an absolute path. Accepts a bare filename or any path.
    """
    candidate = Path(layout).expanduser()
    if not candidate.is_file():
        candidate = REPO_ROOT / "configs" / candidate.name
    if not candidate.is_file():
        raise FileNotFoundError(f"Lead layout not found: {layout}")
    return {LAYOUT_KEY: str(candidate.resolve())}


class DigitizerNotFound(RuntimeError):
    """The Open-ECG-Digitizer checkout could not be located."""


class DigitizerFailed(RuntimeError):
    """The digitizer ran but exited non-zero."""


def digitizer_home() -> Path:
    """Locate the Open-ECG-Digitizer checkout.

    Resolution order: ``$OPEN_ECG_DIGITIZER_HOME``, then a sibling ``Open-ECG-Digitizer``
    directory next to this repo. Raises ``DigitizerNotFound`` with remediation steps.
    """
    # An explicitly set env var is honoured strictly: falling back to a different
    # checkout would silently attribute results to the wrong digitizer version, which
    # matters when a specific checkout has been validated.
    env_value = os.environ.get(ENV_HOME)
    if env_value:
        candidate = Path(env_value).expanduser()
        if (candidate / "src" / "digitize.py").is_file():
            return candidate.resolve()
        raise DigitizerNotFound(
            f"{ENV_HOME} is set to {candidate}, but that is not an Open-ECG-Digitizer "
            f"checkout (no src/digitize.py there).\n\n"
            f"Point it at a real checkout, unset it to use the default sibling location, "
            f"or run: bash scripts/setup_digitizer.sh"
        )

    sibling = REPO_ROOT.parent / "Open-ECG-Digitizer"
    if (sibling / "src" / "digitize.py").is_file():
        return sibling.resolve()

    raise DigitizerNotFound(
        f"Could not find an Open-ECG-Digitizer checkout at {sibling}.\n\n"
        f"Fix it with either:\n"
        f"  export {ENV_HOME}=/path/to/Open-ECG-Digitizer\n"
        f"  bash scripts/setup_digitizer.sh   # clones and patches one for you"
    )


def resolve_mode(mode: str | None = None) -> str:
    """``mode`` if given, else ``$ECG_DIGITIZER_MODE``, else persistent."""
    value = (mode or os.environ.get(ENV_MODE) or MODE_PERSISTENT).strip().lower()
    if value not in MODES:
        raise ValueError(f"Unknown digitizer mode {value!r}; expected one of {', '.join(MODES)} ({ENV_MODE}).")
    return value


def resample_override() -> dict[str, str]:
    """The working-resolution override from ``$ECG_DIGITIZER_RESAMPLE_SIZE``, if set."""
    raw = (os.environ.get(ENV_RESAMPLE_SIZE) or "").strip()
    if not raw:
        return {}
    try:
        size = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_RESAMPLE_SIZE} must be an integer number of pixels, got {raw!r}") from None
    if size < MIN_RESAMPLE_SIZE:
        raise ValueError(f"{ENV_RESAMPLE_SIZE} must be at least {MIN_RESAMPLE_SIZE}, got {size}")
    return {RESAMPLE_KEY: str(size)}


class PersistentDigitizer:
    """One long-lived ``python -m src.serve`` process and the line protocol to talk to it.

    Started on first use. If it dies (killed for memory, a crash in native code), the call
    in flight raises ``DigitizerFailed`` and the next call starts a fresh one.
    """

    def __init__(self, home: Path, python_exe: str, config: Path) -> None:
        self.home = home
        self.python_exe = python_exe
        self.config = config
        self.load_seconds: float | None = None
        self.starts = 0
        self._proc: subprocess.Popen[str] | None = None
        self._stderr: IO[str] | None = None
        self._next_id = 0
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _command(self) -> list[str]:
        return [self.python_exe, "-u", "-m", "src.serve", "--config", str(self.config)]

    def _start(self) -> None:
        self._stderr = tempfile.TemporaryFile(mode="w+", prefix="ecg-digitizer-serve-")
        self._proc = subprocess.Popen(
            self._command(),
            cwd=str(self.home),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            env=threads.child_env(),
        )
        self.starts += 1
        ready = self._read_message()
        if ready.get("event") != "ready":
            raise self._failure(f"Persistent digitizer sent {ready!r} instead of its ready message.")
        self.load_seconds = ready.get("load_seconds")

    def _stderr_tail(self, limit: int = 4000) -> str:
        if self._stderr is None:
            return ""
        self._stderr.seek(0)
        return self._stderr.read()[-limit:]

    def _failure(self, message: str) -> DigitizerFailed:
        """Build the error and discard the process, so the next call starts a new one."""
        tail = self._stderr_tail()
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
            message += f"\n  exit code: {self._proc.returncode}"
            _close_pipes(self._proc)
        self._proc = None
        return DigitizerFailed(
            f"{message}\n  cmd: {' '.join(self._command())}\n  cwd: {self.home}\n\n--- stderr (tail) ---\n{tail}"
        )

    def _read_message(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise self._failure("Persistent digitizer exited unexpectedly.")
        try:
            message = json.loads(line)
        except ValueError:
            raise self._failure(f"Persistent digitizer wrote a line that is not protocol: {line[:200]!r}") from None
        if not isinstance(message, dict):
            raise self._failure(f"Persistent digitizer wrote {message!r}, not an object.")
        return message

    def run(self, overrides: dict[str, str]) -> str:
        """Digitize one request. Returns what the digitizer printed, like its stdout."""
        with self._lock:
            if not self.alive:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            self._next_id += 1
            request_id = self._next_id
            try:
                self._proc.stdin.write(json.dumps({"id": request_id, "overrides": overrides}) + "\n")
                self._proc.stdin.flush()
            except OSError:
                raise self._failure("Persistent digitizer stopped accepting requests.") from None
            reply = self._read_message()
            if reply.get("id") != request_id:
                raise self._failure(f"Persistent digitizer answered request {reply.get('id')!r}, not {request_id}.")
            if not reply.get("ok"):
                raise DigitizerFailed(
                    f"Persistent digitizer failed on the request.\n  overrides: {overrides}\n\n"
                    f"--- error ---\n{reply.get('error')}\n--- log ---\n{reply.get('log', '')}"
                )
            return str(reply.get("log", ""))

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None:
                if proc.poll() is None:
                    try:
                        # EOF on stdin is the server's signal to exit.
                        assert proc.stdin is not None
                        proc.stdin.close()
                        proc.wait(timeout=10)
                    except (OSError, subprocess.TimeoutExpired):
                        proc.kill()
                        proc.wait()
                _close_pipes(proc)
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None


def _close_pipes(proc: subprocess.Popen[str]) -> None:
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


_persistent: PersistentDigitizer | None = None
_persistent_key: tuple[str, ...] | None = None
_persistent_lock = threading.Lock()


def persistent_digitizer(home: Path, python_exe: str, config: Path) -> PersistentDigitizer:
    """The shared persistent digitizer for this checkout, interpreter, config and thread count.

    Only one is kept: a second configuration replaces the first rather than holding two
    sets of models in memory.
    """
    global _persistent, _persistent_key
    # Resolved, so /var and /private/var (macOS) name the same checkout and share one process.
    home, config = Path(home).resolve(), Path(config).resolve()
    key = (str(home), python_exe, str(config), str(threads.resolve_threads()))
    with _persistent_lock:
        if _persistent is None or _persistent_key != key:
            if _persistent is not None:
                _persistent.close()
            _persistent, _persistent_key = PersistentDigitizer(home, python_exe, config), key
        return _persistent


def close_persistent_digitizer() -> None:
    """Stop the shared persistent digitizer, if one is running. Safe to call repeatedly."""
    global _persistent, _persistent_key
    with _persistent_lock:
        if _persistent is not None:
            _persistent.close()
        _persistent, _persistent_key = None, None


atexit.register(close_persistent_digitizer)


def digitize(
    image_dir: str | Path,
    output_dir: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    home: Path | None = None,
    python_exe: str | None = None,
    overrides: dict[str, str] | None = None,
    quiet: bool = False,
    mode: str | None = None,
) -> list[Path]:
    """Digitize every ECG image in ``image_dir``, writing results to ``output_dir``.

    Returns the canonical time-series CSVs produced, sorted by name. Input and output
    paths are passed as ``--overrides`` so a single checked-in config serves every run.
    ``mode`` picks persistent or subprocess execution; see the module docstring.
    """
    home = home or digitizer_home()
    image_dir = Path(image_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    config = Path(config).expanduser().resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not config.is_file():
        raise FileNotFoundError(f"Digitizer config does not exist: {config}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The digitizer resolves the relative paths inside its own config (weights, lead
    # layouts) against the process CWD, so it must run from its checkout root.
    merged = {
        "DATA.images_path": f"{image_dir}/",
        "DATA.output_path": str(output_dir),
    }
    # The deployment's working resolution first, so an explicit caller override still wins.
    merged.update(resample_override())
    merged.update(overrides or {})
    python_exe = python_exe or sys.executable

    if resolve_mode(mode) == MODE_PERSISTENT:
        stdout = persistent_digitizer(home, python_exe, config).run(merged)
    else:
        # Overrides are positional in the digitizer's CLI (KEY=VALUE, after --config).
        cmd = [python_exe, "-m", "src.digitize", "--config", str(config)]
        cmd += [f"{k}={v}" for k, v in merged.items()]

        result = subprocess.run(
            cmd,
            cwd=str(home),
            capture_output=True,
            text=True,
            env=threads.child_env(),
        )
        if result.returncode != 0:
            raise DigitizerFailed(
                f"Digitizer exited {result.returncode}.\n"
                f"  cmd: {' '.join(cmd)}\n  cwd: {home}\n\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )
        stdout = result.stdout

    if not quiet and stdout.strip():
        print(stdout.rstrip())

    return sorted(output_dir.rglob(f"*{CANONICAL_SUFFIX}"))
