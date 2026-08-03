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
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ENV_HOME = "OPEN_ECG_DIGITIZER_HOME"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "digitizer_cpu.yml"

# Filename suffix the digitizer appends to each input image's basename.
CANONICAL_SUFFIX = "_timeseries_canonical.csv"


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


def digitize(
    image_dir: str | Path,
    output_dir: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    home: Path | None = None,
    python_exe: str | None = None,
    overrides: dict[str, str] | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Digitize every ECG image in ``image_dir``, writing results to ``output_dir``.

    Returns the canonical time-series CSVs produced, sorted by name. Input and output
    paths are passed as ``--overrides`` so a single checked-in config serves every run.
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
    merged.update(overrides or {})

    # Overrides are positional in the digitizer's CLI (KEY=VALUE, after --config).
    cmd = [python_exe or sys.executable, "-m", "src.digitize", "--config", str(config)]
    cmd += [f"{k}={v}" for k, v in merged.items()]

    result = subprocess.run(
        cmd,
        cwd=str(home),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DigitizerFailed(
            f"Digitizer exited {result.returncode}.\n"
            f"  cmd: {' '.join(cmd)}\n  cwd: {home}\n\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    if not quiet and result.stdout.strip():
        print(result.stdout.rstrip())

    return sorted(output_dir.rglob(f"*{CANONICAL_SUFFIX}"))
