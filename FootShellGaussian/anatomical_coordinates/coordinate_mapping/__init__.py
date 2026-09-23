"""The shared volumetric coordinate map.

Each shoe's fitted anatomy is reached from the canonical anatomy by following a
velocity field for unit time. Such a map is invertible for any amount of
deformation - integrating the same field backwards is the exact inverse - so
there is no mesh whose cells could fold, and no need to limit how different a
foot may be from the canonical one.

    field     the velocity field and its integrator
    batch     many shoes' fields evaluated in one set of kernels
    build     one map, plus the geometry helpers the rest share
    lookup    canonical addresses: which cell, how deep, how far out
    address   the full address - which point on the foot, and how far out
    cross_shoe  one address sent through every shoe, and read back
    sampling  deterministic area-weighted surface samplers
    audit     injectivity and distortion measured where it is hardest
"""

from .address import (
    Address, AddressBook, CanonicalCorrespondence, FiberTracer, OUTCOME_NAMES,
)
from .batch import BatchedVelocityFields, fit_batch, integrate_batched
from .field import VelocityField, integrate, jacobian_determinants, round_trip_error
from .lookup import CanonicalSemantics

__all__ = [
    "Address",
    "AddressBook",
    "BatchedVelocityFields",
    "CanonicalCorrespondence",
    "CanonicalSemantics",
    "FiberTracer",
    "OUTCOME_NAMES",
    "VelocityField",
    "fit_batch",
    "integrate",
    "integrate_batched",
    "jacobian_determinants",
    "round_trip_error",
]
