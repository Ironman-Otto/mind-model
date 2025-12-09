"""
vector_backend.py

Backend interface for vector similarity. We ship an in-memory backend now and
leave hooks for FAISS/Chroma later without touching the app code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Protocol, Optional
import numpy as np


class VectorBackend(Protocol):
    def add(self, key: str, vector: np.ndarray) -> None: ...
    def get(self, key: str) -> Optional[np.ndarray]: ...
    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]: ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((a @ b) / (na * nb))


@dataclass
class InMemoryVectorBackend:
    """Simplest possible backend: dict + cosine search."""
    store: Dict[str, np.ndarray]

    def __init__(self) -> None:
        self.store = {}

    def add(self, key: str, vector: np.ndarray) -> None:
        self.store[key] = vector.astype(float)

    def get(self, key: str) -> Optional[np.ndarray]:
        return self.store.get(key)

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if query is None:
            return []
        q = query.astype(float)
        scores: List[Tuple[str, float]] = []
        for key, v in self.store.items():
            if v.shape != q.shape:
                continue
            scores.append((key, _cosine(q, v)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# Placeholders for future extension without changing imports
class FaissVectorBackend:  # pragma: no cover – stub
    pass


class ChromaVectorBackend:  # pragma: no cover – stub
    pass
