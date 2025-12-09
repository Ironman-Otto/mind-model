"""
feature_unit.py

Defines FeatureUnit (a functional feature token with optional vector/tensor) and
UnitStore (a light in‑memory registry providing cosine similarity and lookup).

These are modality-agnostic building blocks used by ensembles and concepts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class FeatureUnit:
    """A functional, reusable unit with optional embedding vector.

    Parameters
    ----------
    key : str
        Unique identifier (e.g., "dog.shape_canine").
    modality : str
        Modality tag (e.g., "vision", "language", "audio").
    vector : Optional[np.ndarray]
        Embedding/tensor for similarity and retrieval.
    attributes : Dict[str, str]
        Arbitrary metadata (e.g., {"category": "shape"}).
    """
    key: str
    modality: str
    vector: Optional[np.ndarray] = None
    attributes: Dict[str, str] = field(default_factory=dict)


class UnitStore:
    """A minimal registry mapping unit keys to FeatureUnit objects.

    Exposes cosine similarity over stored vectors when available.
    """
    def __init__(self) -> None:
        self._store: Dict[str, FeatureUnit] = {}

    def add(self, unit: FeatureUnit) -> None:
        """Add or replace a FeatureUnit in the registry."""
        self._store[unit.key] = unit

    def get(self, key: str) -> Optional[FeatureUnit]:
        """Return the FeatureUnit by key, or None if missing."""
        return self._store.get(key)

    def cosine(self, a: str, b: str) -> float:
        """Cosine similarity between vectors of two units (0.0 if unavailable)."""
        ua = self.get(a)
        ub = self.get(b)
        if ua is None or ub is None or ua.vector is None or ub.vector is None:
            return 0.0
        va, vb = ua.vector, ub.vector
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        return float(np.dot(va, vb) / (na * nb)) if na > 0.0 and nb > 0.0 else 0.0
