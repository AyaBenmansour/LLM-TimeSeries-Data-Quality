"""Contracts shared by the scheduler, candidate generators and commit stage."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .reference import Reference, Stats


class Reject(Exception):
    """A valid experiment setting produced an unacceptable candidate."""

    def __init__(self, reason: str, **measurements):
        super().__init__(reason)
        self.reason = reason
        self.measurements = measurements


def check(name: str, condition: bool, **measurements) -> dict:
    if not condition:
        raise Reject(name, **measurements)
    return {"name": name, "passed": True, **measurements}


@dataclass(frozen=True)
class Context:
    family: str
    subtype: str | None
    key: str
    p: dict
    severity: float
    rng: np.random.Generator
    x: np.ndarray
    t: np.ndarray
    idx: np.ndarray
    meta: dict
    stats: "Stats"
    reference: "Reference"
    tolerance: float


@dataclass
class Phase:
    family: str
    subtype: str | None
    idx: np.ndarray
    values: np.ndarray | None = None
    offsets: np.ndarray | None = None
    missing: str | None = None
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    copies: np.ndarray | None = None


@dataclass
class Candidate:
    phases: list[Phase]
    params: dict
    checks: list[dict]


@dataclass(frozen=True)
class InjectionNotFeasible:
    plan_index: int
    family: str
    subtype: str | None
    opportunity: int
    attempts: int
    reason: str
