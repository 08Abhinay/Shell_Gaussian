"""The pipeline's stages, named for what they do.

The research record numbers these stages (the preparation stage, 5, the boundary target stage, the cage deformation stage, the older map lookup,
the older address stage). Those labels say when a thing was built, not what it is, so they are not
used here. The mapping is recorded once, below, for anyone reading the older
reports.

    prepare      normalize the shoe and find its footbed        (was CP 4)
    seat         rest an articulated foot on that footbed       (was CP 5)
    fit          fit foot shape and placement to the cavity     (was CP 6/13)
    anatomy      transfer canonical anatomy to the fitted foot  (was step 14)
    leg          fit the lower leg through the collar           (was step 15)
    join         join foot and leg into one surface             (was step 16)
    coordinates  build the shared volumetric coordinate map     (was the boundary target stage..C)
    address      give every footwear point an anatomical address(was the older address stage)

The output tree is named the same way, so a directory listing reads as the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str
    directory: str
    summary: str
    legacy: str


STAGES: tuple[Stage, ...] = (
    Stage("prepare", "shoes",
          "normalize the shoe, detect the footbed, write the canonical frame",
          "the preparation stage"),
    Stage("seat", "seating",
          "rest an articulated foot on the detected footbed",
          "the seating stage"),
    Stage("fit", "foot",
          "fit foot shape, pose and placement inside the cavity",
          "the accepted containment fit / step 13"),
    Stage("anatomy", "anatomy",
          "transfer canonical anatomical regions onto the fitted foot",
          "step 14"),
    Stage("leg", "leg",
          "fit the lower leg so it leaves through the collar",
          "step 15"),
    Stage("join", "joined",
          "join foot and lower leg into one anatomical surface",
          "step 16"),
    Stage("coordinates", "coordinates",
          "build the shared volumetric coordinate map for each shoe",
          "Checkpoints the boundary target stage, the cage deformation stage, the older map lookup"),
    Stage("address", "addresses",
          "give every footwear surface point an anatomical address",
          "the older address stage"),
)

ORDER = tuple(stage.key for stage in STAGES)
BY_KEY = {stage.key: stage for stage in STAGES}


def slice_stages(first: str, last: str) -> tuple[Stage, ...]:
    if first not in BY_KEY or last not in BY_KEY:
        raise ValueError(f"stage must be one of {ORDER}")
    start, stop = ORDER.index(first), ORDER.index(last)
    if start > stop:
        raise ValueError(f"--from {first} comes after --to {last}")
    return STAGES[start : stop + 1]
