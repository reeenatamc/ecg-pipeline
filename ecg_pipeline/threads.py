"""How many CPU threads torch may use, for the digitizer and for ECGFounder.

Neither model sets a thread count of its own, so each process takes torch's default: one
intra-op thread per physical core it can see. On a shared server that is not a decision
anyone made, and the two stages never run at the same time anyway. ``ECG_TORCH_THREADS``
makes it one.

The digitizer runs in another process (see ``digitizer.py`` for why), so for it the count
travels as ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` in the child's environment, which torch
reads when it starts. ECGFounder runs here, so ``apply_to_torch`` sets it directly before
the model is built.

Deliberately free of a module-level torch import: the digitize-only path must not load it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

ENV_THREADS = "ECG_TORCH_THREADS"

# Upper bound on the default. Measured on the digitizer (docs/RENDIMIENTO.md): past the
# physical cores extra threads only contend, and the target server has 8 vCPUs.
MAX_DEFAULT_THREADS = 8


def available_cpus() -> int:
    """CPUs this process may run on. On Linux this honours a container's cpuset."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined,unused-ignore]
    except AttributeError:  # macOS and Windows have no affinity API
        return os.cpu_count() or 1


def default_threads() -> int:
    return max(1, min(available_cpus(), MAX_DEFAULT_THREADS))


def resolve_threads(env: Mapping[str, str] | None = None) -> int:
    """``ECG_TORCH_THREADS`` if set, else ``default_threads()``. Rejects anything but a positive integer."""
    env = os.environ if env is None else env
    raw = (env.get(ENV_THREADS) or "").strip()
    if not raw:
        return default_threads()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_THREADS} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{ENV_THREADS} must be a positive integer, got {raw!r}")
    return value


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a digitizer process, carrying the resolved thread count."""
    env = os.environ if env is None else env
    out = dict(env)
    threads = str(resolve_threads(env))
    out["OMP_NUM_THREADS"] = threads
    out["MKL_NUM_THREADS"] = threads
    return out


def apply_to_torch() -> int:
    """Set torch's intra-op thread count in this process. Returns the count in effect."""
    import torch

    threads = resolve_threads()
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    return threads
