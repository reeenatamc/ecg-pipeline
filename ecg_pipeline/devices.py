"""Which torch device the analysis runs on, for the digitizer and for ECGFounder alike.

One setting decides it: ``ECG_DEVICE`` (or an explicit ``device=`` argument), with the
values ``cpu`` (the default), ``cuda`` or ``cuda:N``. ECGFounder runs in this process and
takes the device directly. The digitizer runs in another process (see ``digitizer.py``), so
it gets a copy of its config with the two ``device`` keys changed; see
``digitizer.config_for_device``.

A CUDA request is checked before any study starts, so a machine without a usable GPU
fails at startup with a clear message rather than halfway through the first analysis.
``cpu`` is never checked and never imports torch: the digitize-only path must not load it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

ENV_DEVICE = "ECG_DEVICE"
CPU = "cpu"

_VALID = re.compile(r"^(cpu|cuda(:\d+)?)$")


class DeviceError(ValueError):
    """The device setting is not one this pipeline knows how to run on."""


class DeviceUnavailable(RuntimeError):
    """A CUDA device was requested, but this process cannot use it."""


def resolve_device(device: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """``device`` if given, else ``$ECG_DEVICE``, else ``cpu``. Normalized to lower case."""
    env = os.environ if env is None else env
    value = (device or env.get(ENV_DEVICE) or CPU).strip().lower()
    if not _VALID.match(value):
        raise DeviceError(f"Unknown device {value!r}; expected cpu, cuda or cuda:N ({ENV_DEVICE}).")
    return value


def is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def require_device(device: str | None = None) -> str:
    """Resolve the device and make sure this process can actually use it.

    Returns the resolved device. Raises ``DeviceUnavailable`` when CUDA was asked for and
    torch cannot see it (a CPU-only torch build, no driver, no GPU attached, or an index
    past the last GPU).
    """
    resolved = resolve_device(device)
    if not is_cuda(resolved):
        return resolved

    import torch

    if not torch.cuda.is_available():
        build = getattr(torch.version, "cuda", None)
        cause = (
            "this torch build has no CUDA support (install the CUDA wheels)"
            if not build
            else f"torch was built for CUDA {build} but sees no GPU (check the driver and that a GPU is attached)"
        )
        raise DeviceUnavailable(f"{ENV_DEVICE}={resolved} asks for the GPU, but {cause}. Use {ENV_DEVICE}=cpu instead.")

    index = int(resolved.split(":", 1)[1]) if ":" in resolved else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise DeviceUnavailable(f"{ENV_DEVICE}={resolved} asks for GPU {index}, but only {count} GPU(s) are visible.")
    return resolved
