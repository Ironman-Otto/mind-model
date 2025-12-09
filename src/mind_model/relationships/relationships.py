"""
relationships.py

Defines inter-concept relationship edges for semantic links such as IS_A, HAS,
USED_FOR, PART_OF, etc.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass


@dataclass
class RelationshipEdge:
    """Directed edge describing a typed relation to another Concept."""
    relation_type: str
    target_concept_id: uuid.UUID
    description: str = ""
