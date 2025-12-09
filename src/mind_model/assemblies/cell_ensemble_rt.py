"""
cell_ensemble_rt.py

Runtime spatiotemporal ensemble that schedules feature-unit activations over
continuous time and applies Hebbian-like plasticity + decay.

Units are functional feature keys (strings) managed by UnitStore, not neurons.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterable
import math

Time = float
UnitKey = str
Spike = Tuple[Time, UnitKey, float]  # (t, unit_key, strength)


@dataclass
class CellEnsembleRT:
    name: str
    units: Set[UnitKey] = field(default_factory=set)
    context_tags: Set[str] = field(default_factory=set)
    registry: object = None  # Optional: UnitStore for similarity-aware plasticity

    # Schedule: time -> [(unit_key, strength)]
    _scheduled: Dict[Time, List[Tuple[UnitKey, float]]] = field(default_factory=dict, repr=False)
    # Pairwise association weights
    weights: Dict[Tuple[UnitKey, UnitKey], float] = field(default_factory=dict, repr=False)

    # Plasticity params
    eta_hebb: float = 0.02
    decay: float = 0.999
    max_weight: float = 1.0

    # Runtime state
    _t: float = 0.0
    _dt: float = 0.01
    _active_last_step: Set[UnitKey] = field(default_factory=set, repr=False)

    # ---------------- Construction ----------------
    def add_units(self, keys: Iterable[UnitKey]) -> None:
        """Add functional unit keys to the ensemble's membership set."""
        for k in keys:
            self.units.add(k)

    def schedule_spike(self, t: Time, key: UnitKey, strength: float = 1.0) -> None:
        """Schedule a unit activation at absolute time t with strength in [0,1]."""
        if key not in self.units:
            self.units.add(key)
        s = max(0.0, min(1.0, strength))
        self._scheduled.setdefault(float(t), []).append((key, s))

    def schedule_pattern(self, spikes: Iterable[Spike]) -> None:
        """Bulk-schedule many spikes: iterable of (t, key, strength)."""
        for t, k, s in spikes:
            self.schedule_spike(t, k, s)

    # ---------------- Dynamics ---------------------
    def _hebb_increment(self, a: UnitKey, b: UnitKey) -> float:
        """Return a similarity-scaled Hebbian increment for pair (a, b)."""
        base = self.eta_hebb
        if self.registry and hasattr(self.registry, "cosine"):
            sim = max(0.0, float(self.registry.cosine(a, b)))  # 0..1
            return base * (0.5 + 0.5 * sim)
        return base

    def activate_step(self, dt: Optional[float] = None) -> Set[UnitKey]:
        """Advance time by dt, apply scheduled spikes, decay, and Hebbian updates."""
        if dt is None:
            dt = self._dt
        t0, t1 = self._t, self._t + dt
        fired_now: Set[UnitKey] = set()

        to_delete: List[Time] = []
        for ts, events in self._scheduled.items():
            if t0 < ts <= t1:
                for k, s in events:
                    fired_now.add(k)
                to_delete.append(ts)
        for ts in to_delete:
            del self._scheduled[ts]

        # Decay
        if self.weights:
            for k in list(self.weights.keys()):
                self.weights[k] *= self.decay
                if abs(self.weights[k]) < 1e-6:
                    del self.weights[k]

        # Hebbian on co-activated units
        if fired_now:
            fl = list(fired_now)
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    a, b = fl[i], fl[j]
                    key = (a, b) if a < b else (b, a)
                    w = self.weights.get(key, 0.0) + self._hebb_increment(a, b)
                    self.weights[key] = min(w, self.max_weight)

        self._active_last_step = fired_now
        self._t = t1
        return fired_now

    # ---------------- Analysis ---------------------
    def overlap_with(self, other: "CellEnsembleRT") -> float:
        """Jaccard overlap of membership sets (0..1)."""
        if not self.units and not other.units:
            return 1.0
        inter = len(self.units & other.units)
        union = len(self.units | other.units)
        return inter / union if union > 0 else 0.0

    def similarity(self, other: "CellEnsembleRT") -> float:
        """Blend of membership overlap and weight-topology likeness (0..1)."""
        m = self.overlap_with(other)
        keys = set(self.weights.keys()) | set(other.weights.keys())
        if not keys:
            return m
        dot = a2 = b2 = 0.0
        for k in keys:
            wa = self.weights.get(k, 0.0)
            wb = other.weights.get(k, 0.0)
            dot += wa * wb
            a2 += wa * wa
            b2 += wb * wb
        topo = dot / (math.sqrt(a2) * math.sqrt(b2)) if a2 > 0 and b2 > 0 else 0.0
        return 0.6 * m + 0.4 * max(0.0, topo)

    def to_vector(self, window: Tuple[Time, Time], unit_order: Optional[List[UnitKey]] = None) -> List[float]:
        """Counts per unit within [t0, t1] for downstream decoders."""
        t0, t1 = window
        order = unit_order or sorted(self.units)
        counts = {k: 0.0 for k in order}
        for ts, events in self._scheduled.items():
            if t0 <= ts <= t1:
                for k, s in events:
                    if k in counts:
                        counts[k] += s
        return [counts[k] for k in order]
