"""Choosing an ankle pose and shank shape, judged only by the exact evaluator.

The shank has twelve controls: ankle pitch, ankle roll, and ten SUPR shape
coefficients. What matters is that the leg leaves the shoe without passing
through it, and that is decided by an exact triangle-intersection test, not by
a differentiable surrogate - measured against the judge, the surrogate
disagrees precisely at the collar, where the leg lives.

So this searches. The structure is the one the original NumPy fit used, and it
is here because the version that replaced it kept only the first half:

    1. a coarse grid over pitch and roll, shape held neutral
    2. local pose refinement, halving the step
    3. coordinate descent on the shape coefficients
    4. joint refinement - after each shape change, revisit the bend
    5. several starts, so a poor neutral-shape bend cannot trap the shape fit

Step 4 is the one that earns its cost. Shape alone, at a pose chosen for the
neutral shank, saturates well short of what pose and shape reach together:
they trade against each other, and a search that fixes one while moving the
other cannot find that trade.

Nothing here knows what a foot is. ``evaluate`` returns a scored candidate or
``None`` for one that is not anatomically admissible, and everything below is
ordering and bookkeeping - which is why it can be tested without geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class Candidate:
    """One scored shank, and whatever the caller needs to rebuild it."""

    betas: np.ndarray
    pitch: float
    roll: float
    collision_pairs: int
    collision_area_fraction: float
    envelope_maximum: float
    payload: Any = None

    @property
    def beta_norm(self) -> float:
        return float(np.linalg.norm(self.betas))


@dataclass
class SearchReport:
    best: Candidate
    baseline: Candidate
    best_pose_only: Candidate
    evaluated: int
    rejected: int
    stopping_reason: str
    quantum: float
    start_labels: list[str] = field(default_factory=list)
    selected_start: str = "coarse_pose"


def ordering(candidate: Candidate, quantum: float) -> tuple:
    """Preference between two candidates, worst-last.

    Read top to bottom, this says: never leave the frozen envelope; then never
    accept more contact; then, among candidates whose contact is equal to
    within one triangle's worth of area, prefer the most natural shank and the
    gentlest ankle. The quantisation matters - without it a thousandth of a
    triangle of contact would outrank a visibly more natural leg.
    """

    tier = int(np.floor(candidate.collision_area_fraction / quantum + 0.5))
    pose_norm = (candidate.pitch / 20.0) ** 2 + (candidate.roll / 15.0) ** 2
    return (
        candidate.envelope_maximum >= 1.0,
        candidate.collision_pairs > 0,
        tier,
        candidate.beta_norm,
        pose_norm,
        candidate.collision_area_fraction,
        candidate.collision_pairs,
        tuple(np.round(candidate.betas, 8)),
        candidate.pitch,
        candidate.roll,
    )


def search(
    evaluate: Callable[[np.ndarray, float, float], Candidate | None],
    quantum: float,
    *,
    pitch_range: tuple[float, float] = (-20.0, 20.0),
    roll_range: tuple[float, float] = (-15.0, 15.0),
    coarse_step: float = 5.0,
    pose_steps: Sequence[float] = (2.5, 1.25),
    beta_steps: Sequence[float] = (0.5, 0.25),
    joint_steps: Sequence[tuple[float, float]] = ((2.5, 0.5), (1.25, 0.25)),
    joint_rounds: int = 3,
    beta_count: int = 10,
    max_beta_absolute: float = 1.0,
    max_beta_norm: float = 1.5,
    extra_starts: int = 2,
    start_separation: float = 10.0,
    verify: Callable[[Candidate], bool] | None = None,
) -> SearchReport:
    """Find the gentlest natural shank that clears the shoe, if one exists."""

    seen: dict[tuple, Candidate] = {}
    rejected = 0

    def look(betas, pitch, roll) -> Candidate | None:
        nonlocal rejected
        betas = np.asarray(betas, dtype=np.float64)
        pitch = float(np.clip(pitch, *pitch_range))
        roll = float(np.clip(roll, *roll_range))
        if (
            np.max(np.abs(betas)) > max_beta_absolute + 1e-12
            or np.linalg.norm(betas) > max_beta_norm + 1e-12
        ):
            rejected += 1
            return None
        key = tuple(np.round(np.r_[betas, pitch, roll], 8).tolist())
        if key in seen:
            return seen[key]
        candidate = evaluate(betas, pitch, roll)
        if candidate is None:
            rejected += 1
            return None
        seen[key] = candidate
        return candidate

    def better(pool) -> Candidate:
        return min(pool, key=lambda c: ordering(c, quantum))

    def clear(candidate: Candidate) -> bool:
        return candidate.collision_pairs == 0 and candidate.envelope_maximum < 1.0

    zeros = np.zeros(beta_count)
    baseline = look(zeros, 0.0, 0.0)
    if baseline is None:
        raise RuntimeError("the neutral shank was rejected")
    if clear(baseline):
        return SearchReport(baseline, baseline, baseline, len(seen), rejected,
                            "neutral_already_clear", quantum)

    # 1. coarse grid, shape neutral
    grid = [baseline]
    pitches = np.arange(pitch_range[0], pitch_range[1] + 0.5 * coarse_step, coarse_step)
    rolls = np.arange(roll_range[0], roll_range[1] + 0.5 * coarse_step, coarse_step)
    for pitch in pitches:
        for roll in rolls:
            found = look(zeros, float(pitch), float(roll))
            if found is not None:
                grid.append(found)
    pose = better(grid)

    # 2. local pose refinement
    for step in pose_steps:
        pool = [pose]
        for dp, dr in ((-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)):
            found = look(zeros, pose.pitch + dp, pose.roll + dr)
            if found is not None:
                pool.append(found)
        pose = better(pool)
    best_pose_only = pose
    if clear(pose):
        return SearchReport(pose, baseline, pose, len(seen), rejected,
                            "pose_only_clear", quantum)

    # 3. coordinate descent on shape, at that pose
    current = pose
    for step in beta_steps:
        for _ in range(2):
            pool = [current]
            for index in range(beta_count):
                for sign in (-1.0, 1.0):
                    betas = current.betas.copy()
                    betas[index] += sign * step
                    found = look(betas, current.pitch, current.roll)
                    if found is not None:
                        pool.append(found)
            chosen = better(pool)
            if ordering(chosen, quantum) >= ordering(current, quantum):
                break
            current = chosen
            if clear(current):
                break
        if clear(current):
            break

    # 4. joint refinement: bend and shape move against each other, so each has
    #    to be revisited after the other moves.
    def refine(start: Candidate) -> Candidate:
        candidate = start
        for pose_step, beta_step in joint_steps:
            for _ in range(joint_rounds):
                opening = ordering(candidate, quantum)
                pool = [candidate]
                for dp, dr in ((-pose_step, 0.0), (pose_step, 0.0),
                               (0.0, -pose_step), (0.0, pose_step)):
                    found = look(candidate.betas, candidate.pitch + dp,
                                 candidate.roll + dr)
                    if found is not None:
                        pool.append(found)
                candidate = better(pool)
                if clear(candidate):
                    return candidate
                pool = [candidate]
                for index in range(beta_count):
                    for sign in (-1.0, 1.0):
                        betas = candidate.betas.copy()
                        betas[index] += sign * beta_step
                        found = look(betas, candidate.pitch, candidate.roll)
                        if found is not None:
                            pool.append(found)
                candidate = better(pool)
                if clear(candidate):
                    return candidate
                # Stop only when the whole round achieved nothing. Testing the
                # shape step alone ends the round while the bend is still
                # improving, which leaves the fit short of what it can reach.
                if ordering(candidate, quantum) == opening:
                    break
        return candidate

    # 5. several starts, kept far enough apart to explore different bends
    starts: list[tuple[str, Candidate]] = [("coarse_pose", current)]
    for option in sorted(grid, key=lambda c: ordering(c, quantum)):
        if len(starts) > extra_starts:
            break
        if all(
            np.hypot(option.pitch - taken.pitch, option.roll - taken.roll)
            >= start_separation
            for _, taken in starts
        ):
            starts.append(("diverse_pose", option))

    finalists = [("coarse_pose", current)]
    finalists += [(label, refine(seed)) for label, seed in starts]

    # A joined surface that passes through itself is not a fit, however little
    # it touches the shoe. Checked only on the finalists, because it is far
    # more expensive than the collision test.
    if verify is not None:
        safe = [item for item in finalists if verify(item[1])]
        if safe:
            finalists = safe
    label, best = min(finalists, key=lambda item: ordering(item[1], quantum))
    return SearchReport(
        best, baseline, best_pose_only, len(seen), rejected,
        "clear" if clear(best) else "natural_limits_reached_with_contact",
        quantum, [name for name, _ in starts], label,
    )
