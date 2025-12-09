"""
feature_ensemble.py

Feature-level ensembles living inside a Concept (cell assembly). These ensembles
maintain activation, links, decay, and Hebbian updates. They are not the
spatiotemporal runtime ensembles; see cell_ensemble_rt.py for that.
"""
from __future__ import annotations
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureEnsemble:
    """A feature-level ensemble held by a Concept (cell assembly).

    Stores a feature vector, current activation value, and weighted links to
    peer FeatureEnsembles inside the same Concept.
    """
    name: str
    modality: str
    vector: List[float] = field(default_factory=list)
    description: str = ""

    # Runtime state
    activation: float = 0.0

    # Identity and links
    ensemble_id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)
    links: Dict[uuid.UUID, float] = field(default_factory=dict)

    # -------------------------- Utilities --------------------------
    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """Cosine similarity; returns 0.0 if vectors are incompatible."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    def similarity(self, cue_vector: List[float]) -> float:
        """Similarity to an external cue vector, used for direct activation."""
        return self._cosine(self.vector, cue_vector)

    # ----------------------- Structure ops -------------------------
    def add_link(self, target_id: uuid.UUID, weight: float = 0.0) -> None:
        """Create or increment a link to another FeatureEnsemble."""
        self.links[target_id] = self.links.get(target_id, 0.0) + weight

    # ----------------------- Dynamics ops --------------------------
    def decay(self, fraction: float = 0.1) -> None:
        """Leaky decay of activation by a given fraction in [0, 1]."""
        f = max(0.0, min(1.0, fraction))
        self.activation *= (1.0 - f)

    def hebbian(self, coactive_ids: List[uuid.UUID], learning_rate: float = 0.05) -> None:
        """Local Hebbian update for links to co-active ensembles.

        For each ensemble ID in coactive_ids (excluding self), increment link
        weight by learning_rate * self.activation.
        """
        for t in coactive_ids:
            if t == self.ensemble_id:
                continue
            self.links[t] = self.links.get(t, 0.0) + learning_rate * self.activation
