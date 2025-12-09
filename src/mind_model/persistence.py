"""
persistence.py

Persistence helpers to save/load Concepts (engram JSON) and UnitStore contents.
"""
from __future__ import annotations
from typing import Dict, Any
import json
import uuid
import numpy as np
from mind_model.concepts.concept import Concept
from mind_model.concepts.feature_unit import UnitStore, FeatureUnit


# ---------------- Concept Engram ----------------

def save_concept(concept: Concept, path: str) -> None:
    """Write a concept's serialized engram to disk as JSON."""
    data = concept.serialize_engram()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_concept(path: str) -> Concept:
    """Load a concept from a JSON engram file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Concept.from_engram(data)


# ---------------- UnitStore ---------------------

def save_unit_store(store: UnitStore, path: str) -> None:
    """Persist UnitStore to JSON; numpy arrays are converted to lists."""
    blob: Dict[str, Any] = {}
    for k, u in store._store.items():  # noqa: SLF001 – internal access OK for persistence
        vec = u.vector.tolist() if isinstance(u.vector, np.ndarray) else u.vector
        blob[k] = {
            "modality": u.modality,
            "vector": vec,
            "attributes": u.attributes,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)


def load_unit_store(path: str) -> UnitStore:
    """Load UnitStore from JSON (vectors reconstructed as numpy arrays when possible)."""
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    store = UnitStore()
    for k, d in blob.items():
        vec = d.get("vector")
        arr = np.array(vec, dtype=float) if isinstance(vec, list) else None
        store.add(FeatureUnit(key=k, modality=d.get("modality", "unknown"), vector=arr, attributes=d.get("attributes", {})))
    return store
