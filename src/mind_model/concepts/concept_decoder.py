"""
concept_decoder.py

Maps ensemble/assembly activity to symbolic labels. Two decoders provided:
- PopulationThresholdDecoder: label if average activation over named ensembles passes a threshold.
- LinearReadoutDecoder: simple linear layer over concatenated activations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from mind_model.concepts.concept import Concept


@dataclass
class PopulationThresholdDecoder:
    """Rule-based decoder using named ensembles and a threshold per label."""
    label_to_ensembles: Dict[str, List[str]]
    threshold: float = 0.25

    def decode(self, concept: Concept) -> List[Tuple[str, float]]:
        """Return (label, score) for labels that pass the threshold."""
        results: List[Tuple[str, float]] = []
        for label, names in self.label_to_ensembles.items():
            if not names:
                continue
            s = 0.0
            for n in names:
                e = concept.get_ensemble(n)
                if e:
                    s += e.activation
            score = s / max(1, len(names))
            if score >= self.threshold:
                results.append((label, float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


@dataclass
class LinearReadoutDecoder:
    """A tiny linear readout over ensemble activations.

    This is a placeholder for a trainable readout; weights can be learned with
    simple gradient-free updates in small demos or fit offline.
    """
    ensemble_order: List[str]
    W: np.ndarray  # shape: (num_labels, num_ensembles)
    labels: List[str]

    def vectorize(self, concept: Concept) -> np.ndarray:
        """Concatenate activations in the specified order into a vector."""
        v = []
        for n in self.ensemble_order:
            e = concept.get_ensemble(n)
            v.append(e.activation if e else 0.0)
        return np.array(v, dtype=float)

    def decode(self, concept: Concept) -> List[Tuple[str, float]]:
        """Compute scores = W @ v and return sorted (label, score)."""
        v = self.vectorize(concept)
        scores = (self.W @ v.reshape(-1, 1)).flatten()
        out = list(zip(self.labels, [float(s) for s in scores]))
        out.sort(key=lambda x: x[1], reverse=True)
        return out
