"""The shoe set, as one set.

There is no golden/extra distinction here on purpose. The two groups differ
only in when their the preparation and seating stages artifacts were prepared, which is a fact
about this project's history and not about the shoes. ``UNIFIED_INPUT_ROOT``
is a directory of symlinks covering both, so every stage - ours and
``foot_prior``'s - takes exactly one ``--preparation-root``.

``sneaker_vibe`` is the single exclusion: its source geometry has an
inconsistent orientation relative to the accepted normalized set, it is absent
from the processed dataset, and it was excluded from the accepted B3 scope for
the same reason. Excluding it leaves 27, which is the dataset's own count.
"""

from __future__ import annotations

from pathlib import Path

from ..shoe_data import ShoeCase, load_shoe_case


OUTPUT_BASE = Path("/home/ab5298/Outputs/FootShellGaussian")
PIPELINE_ROOT = OUTPUT_BASE / "pipeline27"
UNIFIED_INPUT_ROOT = PIPELINE_ROOT / "inputs"

#: Prepared-artifact roots the unified view is assembled from, in order.
SOURCE_ROOTS = (
    OUTPUT_BASE / "golden_set_evaluation",
    OUTPUT_BASE / "prepared_extra",
)

EXCLUDED = ("sneaker_vibe",)


def build_unified_input_root(
    root: Path = UNIFIED_INPUT_ROOT,
    sources: tuple[Path, ...] = SOURCE_ROOTS,
) -> list[str]:
    """Symlink every prepared shoe into one root. Never copies a dataset."""

    names: list[str] = []
    for kind in ("shoe_preparation", "support_fit"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    for source in sources:
        preparation = source / "shoe_preparation"
        if not preparation.is_dir():
            continue
        for entry in sorted(preparation.iterdir()):
            name = entry.name
            if name in EXCLUDED or not (entry / "shoe_preparation.json").is_file():
                continue
            if not (source / "support_fit" / name).is_dir():
                continue
            for kind in ("shoe_preparation", "support_fit"):
                link = root / kind / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(source / kind / name)
            names.append(name)
    return sorted(names)


def all_shoes(root: Path = UNIFIED_INPUT_ROOT) -> list[str]:
    """Every shoe in the unified set, sorted. Builds the root if absent."""

    preparation = root / "shoe_preparation"
    if not preparation.is_dir():
        return build_unified_input_root(root)
    return sorted(
        item.name
        for item in preparation.iterdir()
        if (item / "shoe_preparation.json").is_file()
    )


def load_case(name: str, root: Path = UNIFIED_INPUT_ROOT) -> ShoeCase:
    return load_shoe_case(name, root)
