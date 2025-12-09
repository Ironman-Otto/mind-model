"""
relations_utils.py

Helpers to inspect and diff Concept relationship edges so the GUI can display
how operations affect the inter-concept graph (including "no changes").
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from mind_model.concepts.concept import Concept

RelationTuple = Tuple[str, str, str]  # (type, target_concept_id_str, description)


def list_relations(concept: Concept) -> List[RelationTuple]:
    """Return this concept's relations as tuples for easy comparison/logging."""
    rows: List[RelationTuple] = []
    for r in concept.relationships:
        rows.append((r.relation_type, str(r.target_concept_id), r.description))
    return rows


def diff_relations(before: List[RelationTuple], after: List[RelationTuple]) -> Dict[str, List[RelationTuple]]:
    """Return added/removed relations as {'added': [...], 'removed': [...]}.

    Inputs are lists of RelationTuple; tuples must match exactly to be considered same.
    """
    set_before = set(before)
    set_after = set(after)
    added = list(set_after - set_before)
    removed = list(set_before - set_after)
    # Sort for stable UI
    added.sort(key=lambda x: (x[0], x[1], x[2]))
    removed.sort(key=lambda x: (x[0], x[1], x[2]))
    return {"added": added, "removed": removed}
