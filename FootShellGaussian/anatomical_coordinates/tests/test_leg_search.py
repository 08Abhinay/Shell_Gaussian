"""The shank search, on objectives whose optimum is chosen in the test.

No geometry here. ``search`` only orders and bookkeeps, so it can be handed a
made-up scoring function whose best answer is known, which is the only way to
tell "the search found the best" from "the search found something".
"""

from __future__ import annotations

import numpy as np
import pytest

from anatomical_coordinates.pipeline.leg_search import Candidate, ordering, search

QUANTUM = 1.0e-4


def _scorer(cost, envelope=0.5, counter=None):
    """Turn a cost function into an ``evaluate`` the search can use."""

    def evaluate(betas, pitch, roll):
        if counter is not None:
            counter.append(1)
        value = max(float(cost(betas, pitch, roll)), 0.0)
        pairs = int(round(value * 1000))
        return Candidate(np.asarray(betas, dtype=np.float64), float(pitch),
                         float(roll), pairs, value, envelope)
    return evaluate


def test_a_clear_neutral_shank_is_taken_immediately():
    calls = []
    report = search(_scorer(lambda b, p, r: 0.0, counter=calls), QUANTUM)
    assert report.stopping_reason == "neutral_already_clear"
    assert report.best.collision_pairs == 0
    assert len(calls) == 1          # nothing else was even tried


def test_pose_alone_is_enough_when_it_is_enough():
    cost = lambda b, p, r: abs(p - 10.0) * 0.001 + abs(r - 5.0) * 0.001
    report = search(_scorer(cost), QUANTUM)
    assert report.stopping_reason == "pose_only_clear"
    assert report.best.collision_pairs == 0
    assert report.best.pitch == pytest.approx(10.0)
    assert report.best.roll == pytest.approx(5.0)
    assert np.allclose(report.best.betas, 0.0)


def test_shape_is_searched_when_pose_cannot_finish():
    """The capability the previous version lacked.

    The objective cannot reach zero at any pose while the shape is neutral,
    so a pose-only search must fail and a search with shape must not.
    """

    def cost(b, p, r):
        return (
            abs(p - 10.0) * 0.001
            + abs(r - 5.0) * 0.001
            + abs(b[0] - 0.5) * 0.01
        )

    report = search(_scorer(cost), QUANTUM)
    assert report.best.collision_pairs == 0, "shape search failed to clear"
    assert report.best.betas[0] == pytest.approx(0.5)
    # and confirm the pose-only stage genuinely could not have done it
    assert report.best_pose_only.collision_pairs > 0
    assert np.allclose(report.best_pose_only.betas, 0.0)


def test_pose_is_revisited_after_shape_moves():
    """Shape and bend trade against each other, so one pass is not enough.

    The bend that suits the neutral shank is not the bend that suits the shape
    the search later picks, so reaching zero needs the bend re-opened *after*
    the shape moves. A search that did pose once and then shape once would
    stop at the shape step, still in contact.
    """

    def cost(b, p, r):
        want_pitch = 10.0 + 10.0 * b[0]      # the best bend moves with shape
        return abs(p - want_pitch) * 0.001 + abs(b[0] - 0.5) * 0.02

    report = search(_scorer(cost), QUANTUM)
    assert report.best.collision_pairs == 0
    assert report.best.betas[0] == pytest.approx(0.5)
    assert report.best.pitch == pytest.approx(15.0)
    # the bend really did have to move away from where the pose stage left it
    assert report.best_pose_only.pitch == pytest.approx(10.0)


def test_shape_is_not_moved_for_an_equal_fit():
    """Ties go to the more natural shank, not to whichever was tried last."""

    def cost(b, p, r):
        want_pitch = 10.0 + 10.0 * b[0]
        return abs(p - want_pitch) * 0.001 + abs(b[0] - 0.5) * 0.01

    report = search(_scorer(cost), QUANTUM)
    assert report.best.beta_norm == pytest.approx(0.0)


def test_natural_shape_limits_are_never_exceeded():
    """A shape the anatomy does not permit is not a solution."""

    # zero cost only at a shape far outside the limits
    cost = lambda b, p, r: abs(b[0] - 5.0) * 0.01
    report = search(_scorer(cost), QUANTUM)
    assert np.max(np.abs(report.best.betas)) <= 1.0 + 1e-9
    assert np.linalg.norm(report.best.betas) <= 1.5 + 1e-9
    assert report.rejected > 0
    assert report.stopping_reason == "natural_limits_reached_with_contact"


def test_ordering_never_prefers_more_contact():
    gentle_but_worse = Candidate(np.zeros(10), 0.0, 0.0, 40, 0.02, 0.5)
    harsh_but_clear = Candidate(np.ones(10) * 0.4, 18.0, 14.0, 0, 0.0, 0.5)
    assert ordering(harsh_but_clear, QUANTUM) < ordering(gentle_but_worse, QUANTUM)


def test_ordering_prefers_the_gentler_of_two_equal_fits():
    quantum = 0.01
    gentle = Candidate(np.zeros(10), 2.0, 1.0, 3, 0.0101, 0.5)
    harsh = Candidate(np.zeros(10), 18.0, 14.0, 3, 0.0104, 0.5)
    # same tier: the ankle decides, not a fourth decimal of area
    assert ordering(gentle, quantum) < ordering(harsh, quantum)


def test_leaving_the_envelope_always_loses():
    outside = Candidate(np.zeros(10), 0.0, 0.0, 0, 0.0, 1.4)
    inside = Candidate(np.zeros(10), 0.0, 0.0, 900, 0.5, 0.6)
    assert ordering(inside, QUANTUM) < ordering(outside, QUANTUM)


def test_finalists_are_verified_before_one_is_chosen():
    """Every finalist is offered to the check, and a rejection is honoured."""

    cost = lambda b, p, r: abs(b[0] - 0.5) * 0.01 + abs(p - 10.0) * 0.001
    seen: list = []

    def watch(candidate):
        seen.append(candidate)
        return True

    report = search(_scorer(cost), QUANTUM, verify=watch)
    assert len(seen) >= 2, "only one finalist was checked"
    assert report.best.collision_pairs == 0


def test_when_nothing_verifies_a_fit_is_still_returned():
    """Refusing every finalist must not leave the shoe without an answer.

    A shank that fails the check is reported, not dropped - the stage that
    reads this is the one equipped to refuse the shoe.
    """

    cost = lambda b, p, r: abs(b[0] - 0.5) * 0.01 + abs(p - 10.0) * 0.001
    report = search(_scorer(cost), QUANTUM, verify=lambda c: False)
    assert report.best is not None


def test_every_candidate_is_scored_only_once():
    calls = []
    cost = lambda b, p, r: abs(p - 3.0) * 0.001 + abs(b[0] - 0.5) * 0.01
    report = search(_scorer(cost, counter=calls), QUANTUM)
    assert report.evaluated == len(calls), "the cache let a duplicate through"
