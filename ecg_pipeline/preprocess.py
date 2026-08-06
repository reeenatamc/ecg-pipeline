"""Prepare ECG images for digitization.

Right now this is one step: upscaling. It is not cosmetic. The digitizer segments the
trace with a U-Net, and on a low-resolution scan the printed trace is thin enough that the
network loses most of it -- the signal comes back fragmented, and everything downstream
(lead coverage, rate, rhythm) is computed from the fragments without any obvious sign that
something went wrong.

Measured on the reference normal ECG (797x446 native, Lanczos x3 to 2391x1338):

    lead I samples recovered      26%  ->  99%
    Einthoven II = I + III (corr)  0.05 ->  0.98
    heart rate                    55 bpm (wrong)  ->  82 bpm (correct)

That is the whole difference between an unusable digitization and a good one, so the
default is to apply it rather than to wait for the caller to ask. What the pipeline does
here is reported per record: the factor never gets applied silently.

Upscaling invents no signal. It resamples the *image* so the segmentation network has
enough pixels to find the trace that was already printed on the paper.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

# Width the factor aims for. The reference ECG failed at 797 px wide and succeeded at 2391
# (x3), so the target sits above the known-bad width and reproduces that x3. Photographs
# from a real ECG machine arrive at ~3000 px and are left untouched.
TARGET_WIDTH = 2000

# Ceiling on the factor, and it is a measurement rather than a guess. On the reference
# right-sided ECG (488 px), x3 -> 1464 px fits the paper grid with a layout matching cost of
# 0.17; x4 -> 1952 px recovers the same 9 leads but the cost degrades to 0.70. More pixels
# stop helping before they stop being produced, so both validated images land on x3.
MAX_FACTOR = 3

# Narrowest upscaled width with evidence behind it (the x3 of that 488 px scan). Below it,
# say so rather than let a fragmented digitization look like a finding.
MIN_USABLE_WIDTH = 1400

# Mirrors DATA.image_extensions in configs/digitizer_cpu.yml. Matched case-insensitively,
# so the '.JPG' the digitizer lists separately needs no special case here.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

AUTO = "auto"
OFF = "off"


class DuplicateRecordName(ValueError):
    """Two input images share a basename, so their outputs would collide."""


def upscale_factor(width: int, target_width: int = TARGET_WIDTH, max_factor: int = MAX_FACTOR) -> int:
    """Smallest integer factor bringing ``width`` up to ``target_width``, capped.

    Integer rather than fractional: it keeps the resampling grid aligned with the pixel
    grid, which matters for an image whose information is thin ruled lines.
    """
    if width <= 0:
        return 1
    return max(1, min(math.ceil(target_width / width), max_factor))


def find_images(image_dir: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[Path]:
    """Input images under ``image_dir``, sorted, recursive.

    Recursive because the digitizer walks its input tree, and staging only the top level
    would silently drop every image in a subdirectory.
    """
    directory = Path(image_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    lowered = tuple(e.lower() for e in extensions)
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in lowered)


def prepare_images(
    image_dir: str | Path,
    work_dir: str | Path,
    upscale: str | int = AUTO,
    target_width: int = TARGET_WIDTH,
    max_factor: int = MAX_FACTOR,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> dict[str, dict[str, Any]]:
    """Populate ``work_dir`` with the images to digitize, upscaled where needed.

    ``upscale`` is ``"auto"`` (factor per image from its width), ``"off"``, or an explicit
    integer factor applied to every image.

    The directory tree is mirrored into ``work_dir`` and each file keeps its basename,
    because the digitizer names its output after the input's path relative to its input root
    and the rest of the pipeline joins on that. Returns one record per image, keyed by that
    relative path without its extension (``"ecg"``, or ``"batch-3/ecg"`` when nested).
    """
    if upscale == OFF:
        factor_for = lambda _w: 1  # noqa: E731
    elif upscale == AUTO:
        factor_for = lambda w: upscale_factor(w, target_width, max_factor)  # noqa: E731
    else:
        fixed = int(upscale)
        if fixed < 1:
            raise ValueError(f"Upscale factor must be >= 1, got {fixed}")
        factor_for = lambda _w: fixed  # noqa: E731

    # Imported here rather than at module scope so that --digitize-only on an already
    # prepared directory, and the test suite, do not depend on Pillow being installed.
    from PIL import Image

    root = Path(image_dir).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[str, dict[str, Any]] = {}
    for source in find_images(image_dir, extensions):
        relative = source.relative_to(root)
        name = relative.with_suffix("").as_posix()
        if name in prepared:
            raise DuplicateRecordName(
                f"{source.name} and {Path(prepared[name]['source']).name} sit in the same "
                f"directory under the basename {name!r}; the digitizer would write both to "
                f"the same output files. Rename one of them."
            )
        destination_dir = work_dir / relative.parent
        destination_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(source) as image:
            width, height = image.size
            factor = factor_for(width)
            if factor == 1:
                # Symlinked, not copied: a batch of phone photographs is hundreds of MB and
                # nothing here needs a second copy of it. Falls back where symlinks are not
                # available. The digitizer only ever reads its input.
                destination = destination_dir / source.name
                try:
                    destination.symlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
                size = (width, height)
            else:
                # PNG, not the source format: an upscale followed by a JPEG re-encode would
                # stamp compression artefacts onto the trace the U-Net is about to segment.
                destination = destination_dir / f"{relative.stem}.png"
                size = (width * factor, height * factor)
                # Convert before resizing -- resampling a palette image interpolates palette
                # indices, which is meaningless.
                image.convert("RGB").resize(size, Image.LANCZOS).save(destination)

        prepared[name] = {
            "source": str(source),
            "prepared": str(destination),
            "original_size": [width, height],
            "size": [size[0], size[1]],
            "upscale_factor": factor,
        }

    return prepared


def preprocessing_warnings(record: dict[str, Any] | None) -> list[str]:
    """Warnings about the image the digitizer was actually given."""
    if not record:
        return []

    warnings: list[str] = []
    width = record["original_size"][0]
    factor = record["upscale_factor"]

    # Against MIN_USABLE_WIDTH rather than TARGET_WIDTH: a 488 px scan upscaled to 1464 px
    # digitizes well, so warning at everything short of the 2000 px target would cry wolf on
    # an input that works.
    if factor > 1 and width * factor < MIN_USABLE_WIDTH:
        warnings.append(
            f"The image is {width} px wide; even at the maximum x{factor} upscale it reaches "
            f"only {width * factor} px, under the {MIN_USABLE_WIDTH} px this pipeline has "
            f"evidence for. Digitization is likely to be fragmented. Re-scan at a higher "
            f"resolution."
        )
    return warnings
