"""
concept.py

Concept == Cell Assembly (structural substrate). Coordinates FeatureEnsembles
and supports stimulation, lateral inhibition, Hebbian learning, relationships,
and engram I/O.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import uuid
from mind_model.concepts.feature_ensemble import FeatureEnsemble
from mind_model.relationships.relationships import RelationshipEdge


class Concept:
    """A structural cell assembly coordinating intra-assembly ensembles."""

    def __init__(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        inhibition_gain: float = 0.25,
        activation_threshold: float = 0.15,
    ) -> None:
        self.concept_id: uuid.UUID = uuid.uuid4()
        self.name: str = name
        self.description: str = description
        self.metadata: Dict[str, Any] = metadata or {"version": "3.0"}

        # Ensembles and lookup tables
        self.ensembles_by_id: Dict[uuid.UUID, FeatureEnsemble] = {}
        self.ensembles_by_name: Dict[str, uuid.UUID] = {}

        # Inter-concept relationships
        self.relationships: List[RelationshipEdge] = []

        # Dynamics params
        self.inhibition_gain: float = inhibition_gain
        self.activation_threshold: float = activation_threshold

    # ---------------- Ensemble management ----------------
    def add_ensemble(self, ensemble: FeatureEnsemble) -> None:
        """Register a FeatureEnsemble with this Concept."""
        self.ensembles_by_id[ensemble.ensemble_id] = ensemble
        self.ensembles_by_name[ensemble.name] = ensemble.ensemble_id

    def get_ensemble(self, name: str) -> Optional[FeatureEnsemble]:
        """Retrieve a FeatureEnsemble by its name."""
        e_id = self.ensembles_by_name.get(name)
        return self.ensembles_by_id.get(e_id) if e_id else None

    # ---------------- Activation & inhibition --------------
    def _lateral_inhibition(self) -> None:
        """Divisive normalization across all ensembles to enforce competition."""
        total = sum(max(0.0, e.activation) for e in self.ensembles_by_id.values())
        if total <= 1e-9:
            return
        denom = 1.0 + self.inhibition_gain * total
        for e in self.ensembles_by_id.values():
            e.activation = e.activation / denom

    def decay_all(self, fraction: float = 0.1) -> None:
        """Apply activation decay to all ensembles."""
        for e in self.ensembles_by_id.values():
            e.decay(fraction=fraction)

    # ---------------- Learning -----------------------------
    def learn_hebbian(self, learning_rate: float = 0.05, min_activation: Optional[float] = None) -> None:
        """Strengthen links among co-active ensembles using a Hebbian rule."""
        thr = self.activation_threshold if min_activation is None else min_activation
        coactive_ids = [eid for eid, e in self.ensembles_by_id.items() if e.activation >= thr]
        for e in self.ensembles_by_id.values():
            if e.activation >= thr:
                e.hebbian(coactive_ids=coactive_ids, learning_rate=learning_rate)

    # ------------- Stimulation & completion ----------------
    def stimulate(self, cues: Dict[str, List[float]], gain: float = 1.0) -> Dict[str, float]:
        """Provide partial cues and perform one activation step.

        Steps
        -----
        1) Direct activation from cues by cosine similarity
        2) One-step spread along intra-assembly links
        3) Lateral inhibition
        """
        # 1) Direct activation
        for name, vec in cues.items():
            ens = self.get_ensemble(name)
            if ens:
                ens.activation += gain * ens.similarity(vec)

        # 2) One-step spread
        delta: Dict[uuid.UUID, float] = {eid: 0.0 for eid in self.ensembles_by_id}
        for s_id, s in self.ensembles_by_id.items():
            if s.activation <= 0.0:
                continue
            for t_id, w in s.links.items():
                delta[t_id] = delta.get(t_id, 0.0) + s.activation * w
        for t_id, d in delta.items():
            self.ensembles_by_id[t_id].activation += d

        # 3) Normalize
        self._lateral_inhibition()

        return {ens.name: round(ens.activation, 4) for ens in self.ensembles_by_id.values()}

    def recall_partial(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k most active ensembles after a stimulate() call."""
        items = [(e.name, e.activation) for e in self.ensembles_by_id.values()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

    # ------------- Inter-concept graph ---------------------
    def add_relationship(self, relation_type: str, target_concept_id: uuid.UUID, description: str = "") -> None:
        """Attach a labeled semantic relationship to another concept."""
        self.relationships.append(RelationshipEdge(relation_type, target_concept_id, description))

    # ------------- Engram I/O ------------------------------
    def serialize_engram(self) -> Dict[str, Any]:
        """Serialize the concept with ensembles and intra-assembly links."""
        return {
            "concept_id": str(self.concept_id),
            "name": self.name,
            "description": self.description,
            "ensembles": [
                {
                    "ensemble_id": str(e.ensemble_id),
                    "name": e.name,
                    "modality": e.modality,
                    "vector": e.vector,
                    "links": {str(t): w for t, w in e.links.items()},
                }
                for e in self.ensembles_by_id.values()
            ],
            "relationships": [
                {
                    "type": r.relation_type,
                    "target_concept_id": str(r.target_concept_id),
                    "description": r.description,
                }
                for r in self.relationships
            ],
            "params": {
                "inhibition_gain": self.inhibition_gain,
                "activation_threshold": self.activation_threshold,
            },
            "metadata": self.metadata,
        }

    @staticmethod
    def from_engram(data: Dict[str, Any]) -> "Concept":
        """Reconstruct a Concept from a serialized engram dict."""
        c = Concept(
            name=data.get("name", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            inhibition_gain=data.get("params", {}).get("inhibition_gain", 0.25),
            activation_threshold=data.get("params", {}).get("activation_threshold", 0.15),
        )
        c.concept_id = uuid.UUID(data["concept_id"])

        # Ensembles
        for ed in data.get("ensembles", []):
            e = FeatureEnsemble(
                name=ed["name"],
                modality=ed.get("modality", "unknown"),
                vector=ed.get("vector", []),
            )
            e.ensemble_id = uuid.UUID(ed["ensemble_id"])  # preserve
            c.add_ensemble(e)

        # Links
        for ed in data.get("ensembles", []):
            s_id = uuid.UUID(ed["ensemble_id"])
            s = c.ensembles_by_id[s_id]
            for t_str, w in ed.get("links", {}).items():
                s.links[uuid.UUID(t_str)] = w

        # Relationships
        for rd in data.get("relationships", []):
            c.relationships.append(
                RelationshipEdge(
                    rd["type"], uuid.UUID(rd["target_concept_id"]), rd.get("description", "")
                )
            )
        return c
