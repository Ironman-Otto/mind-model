"""
graph_view.py

Build a PyVis network from selected Concept objects and render into Streamlit
via st.components.v1.html. Nodes are concepts, edges are relationships.
"""
from __future__ import annotations
from typing import Dict, List
from pyvis.network import Network
from mind_model.concepts.concept import Concept


def build_graph(concepts: Dict[str, Concept], physics: bool = True) -> Network:
    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.barnes_hut() if physics else net.hrepulsion()

    # Map concept_id -> label
    id_to_label: Dict[str, str] = {}
    for label, c in concepts.items():
        node_id = str(c.concept_id)
        id_to_label[node_id] = label
        title = f"<b>{label}</b><br/>{c.description}<br/>ensembles: {len(c.ensembles_by_id)}"
        net.add_node(node_id, label=label, title=title)

    # Add edges
    for label, c in concepts.items():
        src_id = str(c.concept_id)
        for r in c.relationships:
            dst_id = str(r.target_concept_id)
            # Show label for destination if we have it; otherwise keep raw id
            dst_label = id_to_label.get(dst_id, dst_id[:8])
            net.add_edge(src_id, dst_id, label=r.relation_type)

    return net
