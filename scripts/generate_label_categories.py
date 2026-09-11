#!/usr/bin/env python3
"""Generate ``ecg_pipeline/label_categories.py`` from ``docs/etiquetas_es_borrador.csv``.

The CSV is the source of truth for which of the 10 clinical categories each of the
model's 150 labels belongs to (see ``docs/etiquetas_es_borrador.md`` for how the mapping
was drafted). Regenerate the module after editing the CSV rather than hand-patching the
generated dict, the same way app-EKG's ``src/constants/labelsEs.ts`` is kept in sync with
this file's Spanish-translation columns:

    python scripts/generate_label_categories.py

The generated module is committed, not built at import time, so the pipeline never reads
the CSV at runtime and a checkout without ``docs/`` still imports cleanly.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "docs" / "etiquetas_es_borrador.csv"
OUTPUT_PATH = REPO_ROOT / "ecg_pipeline" / "label_categories.py"

# The 10 clinical categories a label can be filed under. Fixed, English slugs: the app
# groups observations by these, so they are not translated or renamed here.
VALID_CATEGORIES = {
    "ritmo",
    "conduccion",
    "repolarizacion",
    "isquemia_infarto",
    "marcapasos",
    "eje",
    "hipertrofia",
    "tecnico",
    "resumen",
    "otro",
}

HEADER = '''"""Clinical category for each of the model's 150 labels.

GENERATED FILE. Do not edit by hand -- edit ``docs/etiquetas_es_borrador.csv`` and rerun::

    python scripts/generate_label_categories.py

``docs/etiquetas_es_borrador.csv`` is the source of truth; see ``docs/etiquetas_es_borrador.md``
for how each label was assigned to a category. The category slugs are fixed and are never
translated or renamed: the app groups observations by them.
"""

from __future__ import annotations

DEFAULT_CATEGORY = "otro"

LABEL_CATEGORIES: dict[str, str] = {
'''

FOOTER = "}\n"


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render(rows: list[dict[str, str]]) -> str:
    lines = [HEADER]
    for row in rows:
        label, category = row["label_en"], row["categoria"]
        if category not in VALID_CATEGORIES:
            raise ValueError(f"{label!r} has an unknown category {category!r}")
        lines.append(f"    {label!r}: {category!r},\n")
    lines.append(FOOTER)
    return "".join(lines)


def main() -> None:
    rows = load_rows()
    labels = [row["label_en"] for row in rows]
    if len(labels) != len(set(labels)):
        seen: set[str] = set()
        dupes = sorted({label for label in labels if label in seen or seen.add(label)})  # type: ignore[func-returns-value]
        print(f"Duplicate label_en values in {CSV_PATH}: {dupes}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} labels to {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
