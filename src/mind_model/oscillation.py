"""
oscillation.py

Oscillation gates to organize multi-step activation (theta/gamma-like packets).
These utilities can be used by drivers to coordinate stimulate()/learn() calls.
"""
from __future__ import annotations
from typing import Iterable, List, Callable
import math

PhaseCallback = Callable[[float], None]


def phase_sequence(total_time: float, theta_hz: float, gamma_per_theta: int) -> List[float]:
    """Return phase times (in seconds) for a theta cycle subdivided into gamma packets.

    Example: total_time=1.0, theta_hz=5, gamma_per_theta=4 -> ~5 theta cycles,
    each split into 4 gamma packet timestamps.
    """
    phases: List[float] = []
    if theta_hz <= 0 or gamma_per_theta <= 0 or total_time <= 0:
        return phases
    theta_period = 1.0 / theta_hz
    t = 0.0
    while t < total_time:
        gamma_dt = theta_period / float(gamma_per_theta)
        for i in range(gamma_per_theta):
            phases.append(t + (i + 1) * gamma_dt)
        t += theta_period
    return phases


def run_phased(total_time: float, theta_hz: float, gamma_per_theta: int, on_phase: PhaseCallback) -> None:
    """Invoke `on_phase(t)` at each gamma packet time to gate updates.

    The callback could perform: concept.stimulate(...), concept.learn_hebbian(...), etc.
    """
    for t in phase_sequence(total_time, theta_hz, gamma_per_theta):
        on_phase(t)
