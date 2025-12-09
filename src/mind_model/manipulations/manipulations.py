"""
manipulations.py (v2)

Concept operations now return (result_concept | None, relation_deltas, notes)
so the GUI can show the inter-concept relationship effects for every action.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy
import uuid  # ← Add this line
from mind_model.concepts.concept import Concept
from mind_model.concepts.feature_ensemble import FeatureEnsemble
from mind_model.utils.relations_utils import list_relations, diff_relations


Result = Tuple[Optional[Concept], Dict[str, List[Tuple[str, str, str]]], str]


# ----------------------- Helpers -----------------------

def _ensemble_name_set(c: Concept) -> Set[str]:
    return set(c.ensembles_by_name.keys())


def _copy_concept_shallow(c: Concept, new_name: str) -> Concept:
    nc = Concept(
        name=new_name,
        description=f"Derived from {c.name}",
        metadata=deepcopy(c.metadata),
        inhibition_gain=c.inhibition_gain,
        activation_threshold=c.activation_threshold,
    )
    return nc


# --------------------- Operations ----------------------

def compare_concepts(a: Concept, b: Concept) -> Result:
    """Return structural/semantic metrics; no new concept is created."""
    from math import sqrt

    na = _ensemble_name_set(a)
    nb = _ensemble_name_set(b)
    inter = na & nb
    union = na | nb
    jaccard = len(inter) / len(union) if union else 1.0

    def cosine(u: List[float], v: List[float]) -> float:
        if not u or not v or len(u) != len(v):
            return 0.0
        dot = sum(x * y for x, y in zip(u, v))
        nu = sqrt(sum(x * x for x in u))
        nv = sqrt(sum(y * y for y in v))
        return 0.0 if nu == 0.0 or nv == 0.0 else dot / (nu * nv)

    cos_vals: List[float] = []
    for n in inter:
        ea = a.get_ensemble(n)
        eb = b.get_ensemble(n)
        if ea and eb:
            cos_vals.append(cosine(ea.vector, eb.vector))
    mean_cos = sum(cos_vals) / len(cos_vals) if cos_vals else 0.0

    def avg_link_density(c: Concept) -> float:
        if not c.ensembles_by_id:
            return 0.0
        s = 0.0
        for e in c.ensembles_by_id.values():
            s += float(len(e.links))
        return s / float(len(c.ensembles_by_id))

    link_density_diff = abs(avg_link_density(a) - avg_link_density(b))

    notes = (
        f"Compare: jaccard={jaccard:.3f}, mean_vector_cosine={mean_cos:.3f}, "
        f"link_density_diff={link_density_diff:.3f}"
    )
    # No relation change on compare
    return None, {"added": [], "removed": []}, notes


def merge_concepts(a: Concept, b: Concept, new_name: str = "Merged") -> Result:
    """Union of ensembles and links; now also links the result back to A and B."""
    before_rels = []
    c = _copy_concept_shallow(a, new_name)

    # Copy all ensembles from A
    for ea in a.ensembles_by_id.values():
        na = FeatureEnsemble(
            name=ea.name, modality=ea.modality,
            vector=list(ea.vector), description=ea.description
        )
        c.add_ensemble(na)

    # Copy all ensembles from B (resolve name collisions)
    for eb in b.ensembles_by_id.values():
        name_b = eb.name if eb.name not in c.ensembles_by_name else f"{eb.name}__B"
        nb = FeatureEnsemble(
            name=name_b, modality=eb.modality,
            vector=list(eb.vector), description=eb.description
        )
        c.add_ensemble(nb)

    # Copy intra-ensemble links within each source block
    name_to_id = {e.name: eid for eid, e in c.ensembles_by_id.items()}

    def copy_links(src: Concept, suffix: str = "") -> None:
        for e in src.ensembles_by_id.values():
            sname = e.name if suffix == "" else (e.name + suffix)
            sid = name_to_id.get(sname)
            if sid is None:
                continue
            se = c.ensembles_by_id[sid]
            for tid, w in e.links.items():
                tname = src.ensembles_by_id[tid].name
                tname2 = tname if suffix == "" else (tname + suffix)
                tid2 = name_to_id.get(tname2)
                if tid2 is not None:
                    se.links[tid2] = w

    copy_links(a, suffix="")
    copy_links(b, suffix="__B")

    # NEW: link merged concept back to its sources
    c.add_relationship("MERGED_FROM", a.concept_id, description=f"source:{a.name}")
    c.add_relationship("MERGED_FROM", b.concept_id, description=f"source:{b.name}")

    after_rels = list_relations(c)
    rel_delta = {"added": after_rels, "removed": []}
    notes = f"Merged {a.name} and {b.name} into {c.name}; added MERGED_FROM edges."
    return c, rel_delta, notes

def intersect_concepts(a: Concept, b: Concept, new_name: str = "Intersect") -> Result:
    """
    Create a new concept that represents the intersection of A and B.

    Behavior:
    - Keeps only ensembles present in both A and B (matched by ensemble *name*).
    - Preserves intra-assembly links where both endpoints survive.
    - NEW: Adds any shared relationships that A and B both have to the same target
      with the same relation_type (e.g., Dog IS_A Animal and Cat IS_A Animal).

    Returns:
        (result_concept, relation_deltas, notes)
    """
    # ---- 1) Intersect ensembles by name and copy them into a new concept ----
    result = _copy_concept_shallow(a, new_name)
    shared_names = _ensemble_name_set(a) & _ensemble_name_set(b)

    # Map original ensemble names -> new ensemble IDs in the result
    name_to_new_id = {}
    for n in shared_names:
        ea = a.get_ensemble(n)
        if not ea:
            continue
        ne = FeatureEnsemble(
            name=ea.name,
            modality=ea.modality,
            vector=list(ea.vector),
            description=ea.description,
        )
        result.add_ensemble(ne)
        name_to_new_id[n] = ne.ensemble_id

    # Recreate links where both endpoints are still present
    for n in shared_names:
        ea = a.get_ensemble(n)
        if not ea:
            continue
        se = result.get_ensemble(n)
        if not se:
            continue
        for t_id, w in ea.links.items():
            t_name = a.ensembles_by_id[t_id].name
            if t_name in name_to_new_id:
                se.links[name_to_new_id[t_name]] = w

    # ---- 2) Intersect relationships (A and B share same relation_type + target) ----
    # Represent each relation as (type, target_uuid_str) for set math
    rels_a = {(r.relation_type, str(r.target_concept_id)) for r in a.relationships}
    rels_b = {(r.relation_type, str(r.target_concept_id)) for r in b.relationships}
    shared_rels = rels_a & rels_b

    # Add shared relations to the *new* intersect concept
    for rel_type, target_id_str in shared_rels:
        # Store a short note so the GUI can show provenance
        result.add_relationship(rel_type, uuid.UUID(target_id_str), description="shared relation")

    # ---- 3) Report relation deltas and notes for the GUI ----
    rel_delta = {"added": list_relations(result), "removed": []}
    notes = (
        f"Intersection created. Shared ensembles: {len(shared_names)}. "
        f"Shared relations added: {len(shared_rels)}."
    )
    return result, rel_delta, notes



def subtract_concepts(a: Concept, b: Concept, new_name: str = "A_minus_B") -> Result:
    c = _copy_concept_shallow(a, new_name)
    nb = _ensemble_name_set(b)

    keep = [e for e in a.ensembles_by_id.values() if e.name not in nb]
    name_to_new = {}
    for e in keep:
        ne = FeatureEnsemble(name=e.name, modality=e.modality, vector=list(e.vector), description=e.description)
        c.add_ensemble(ne)
        name_to_new[e.name] = ne.ensemble_id

    for e in keep:
        se = c.get_ensemble(e.name)
        if not se:
            continue
        for tid, w in e.links.items():
            tname = a.ensembles_by_id[tid].name
            if tname in name_to_new:
                se.links[name_to_new[tname]] = w

    rel_delta = {"added": [], "removed": []}
    notes = "Subtraction created. Relations unchanged (derived concept has none by default)."
    return c, rel_delta, notes


def bind_relation(a: Concept, b: Concept, relation_type: str, description: str = "") -> Result:
    before = list_relations(a)
    # mutate a by adding a relation to b
    a.add_relationship(relation_type, b.concept_id, description=description)
    after = list_relations(a)
    delta = diff_relations(before, after)
    notes = f"Added relation {relation_type} from {a.name} to {b.name}."
    return a, delta, notes
