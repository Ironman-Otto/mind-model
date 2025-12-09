# gui_app_streamlit.py (v3) – only changed bits shown; replace your file if easier

from __future__ import annotations
import streamlit as st
from typing import List, Dict
from mind_model.concepts.seed_concepts import list_catalog
from mind_model.concepts.concept import Concept
from mind_model.concepts.feature_ensemble import FeatureEnsemble
from mind_model.manipulations.manipulations import (
    compare_concepts, merge_concepts,
    intersect_concepts, subtract_concepts, bind_relation,
)
from mind_model.utils.relations_utils import list_relations
from mind_model.graphs.graph_view import build_graph
import streamlit.components.v1 as components

st.set_page_config(page_title="Concept Manipulation Demo", layout="wide")
st.title("Concept Manipulation Demo")
st.caption("Create concepts and apply operations to show structural/functional reasoning.")

# ---------------- Workspace init ----------------
def _init_workspace() -> Dict[str, Concept]:
    base = list_catalog()            # Animal, Dog, Cat, Car
    return {k: v for k, v in base.items()}

if "workspace" not in st.session_state:
    st.session_state.workspace = _init_workspace()

workspace: Dict[str, Concept] = st.session_state.workspace

# ---------------- Sidebar: Setup ----------------
st.sidebar.header("Setup")
num_concepts = st.sidebar.selectbox("Number of concepts", options=[2, 3], index=0)
choices = list(workspace.keys())  # <-- use workspace, not static catalog

selected_labels: List[str] = []
for i in range(num_concepts):
    lbl = st.sidebar.selectbox(
        f"Select concept {i+1}",
        options=choices,
        index=min(i, len(choices)-1),
        key=f"pick_{i}"
    )
    selected_labels.append(lbl)

instances: List[Concept] = [workspace[name] for name in selected_labels]

# ---------------- Tabs ----------------
TAB_INSPECT, TAB_OPERATE, TAB_GRAPH, TAB_BUILD = st.tabs(["Inspect", "Operate", "Graph", "Build"])

with TAB_INSPECT:
    st.subheader("Selected Concepts")
    cols = st.columns(len(instances))
    for i, c in enumerate(instances):
        with cols[i]:
            st.markdown(f"**{c.name}** – {c.description}")
            st.write("Ensembles:")
            items = []
            for eid, e in c.ensembles_by_id.items():
                items.append({
                    "name": e.name, "modality": e.modality,
                    "activation": round(e.activation, 4), "links": len(e.links),
                })
            st.dataframe(items, hide_index=True, use_container_width=True)
            st.write("Relations:")
            rows = [{"type": r[0], "target": r[1][:8], "desc": r[2]} for r in list_relations(c)]
            st.dataframe(rows, hide_index=True, use_container_width=True)

with TAB_OPERATE:
    st.subheader("Operations")
    operation = st.selectbox(
        "Choose an operation",
        ("Compare (two concepts)", "Merge (A ∪ B)", "Intersect (A ∩ B)",
         "Subtract (A − B)", "Bind Relation (A —rel→ B)")
    )
    colA, colB = st.columns(2)
    with colA:
        idx_a = st.selectbox("Pick A", options=list(range(len(instances))),
                             format_func=lambda i: instances[i].name, key="op_A")
    with colB:
        idx_b = st.selectbox("Pick B", options=list(range(len(instances))),
                             index=min(1, len(instances)-1),
                             format_func=lambda i: instances[i].name, key="op_B")
    cA = instances[idx_a]; cB = instances[idx_b]

    rel_input = "RELATED_TO"
    if operation.startswith("Bind Relation"):
        rel_input = st.text_input("Relation type (e.g., IS_A, HAS, USED_FOR)", value="RELATED_TO")

    add_to_workspace = st.checkbox("Add result to workspace", value=True)

    if st.button("Run Operation", type="primary"):
        result = None; rels = {"added": [], "removed": []}; notes = ""
        if operation.startswith("Compare"):
            result, rels, notes = compare_concepts(cA, cB)
        elif operation.startswith("Merge"):
            result, rels, notes = merge_concepts(cA, cB, new_name=f"{cA.name}_merged_{cB.name}")
        elif operation.startswith("Intersect"):
            result, rels, notes = intersect_concepts(cA, cB, new_name=f"{cA.name}_inter_{cB.name}")
        elif operation.startswith("Subtract"):
            result, rels, notes = subtract_concepts(cA, cB, new_name=f"{cA.name}_minus_{cB.name}")
        elif operation.startswith("Bind Relation"):
            result, rels, notes = bind_relation(cA, cB, relation_type=rel_input,
                                                description="GUI-created relation")

        st.success("Operation complete.")
        st.write(notes)
        st.write("Relationship changes:")
        st.json(rels)

        if result and add_to_workspace:
            # Handle name collision by uniquifying
            base = result.name; suffix = 1
            name = base
            while name in workspace:
                suffix += 1
                name = f"{base}_{suffix}"
            result.name = name
            workspace[name] = result
            st.info(f"Added '{name}' to workspace. It will appear in pickers and graph.")

with TAB_GRAPH:
    st.subheader("Concept Graph (workspace)")
    net = build_graph(workspace)
    net_html = net.generate_html(notebook=False)
    components.html(net_html, height=620, scrolling=True)
    st.caption("Nodes are concepts; edges are relationships. Workspace includes results you added.")

with TAB_BUILD:
    st.subheader("Create a Concept")
    with st.form("build_concept"):
        concept_name = st.text_input("Name", value="NewConcept")
        concept_desc = st.text_input("Description", value="User-defined concept")

        # Feature builder
        st.markdown("**Features (ensembles)**")
        num_feats = st.number_input("Number of features", min_value=0, max_value=20, value=2, step=1)
        feats = []
        for i in range(int(num_feats)):
            st.markdown(f"_Feature {i+1}_")
            fname = st.text_input(f"Name {i+1}", key=f"f_name_{i}", value=f"feat_{i+1}")
            fmod = st.text_input(f"Modality {i+1}", key=f"f_mod_{i}", value="vision")
            fvec = st.text_input(f"Vector (comma-separated) {i+1}", key=f"f_vec_{i}", value="1,0,0,0")
            feats.append((fname, fmod, fvec))

        # Relationship builder
        st.markdown("**Relationships**")
        num_rels = st.number_input("Number of relationships", min_value=0, max_value=20, value=0, step=1)
        rels = []
        existing_names = list(workspace.keys())
        for i in range(int(num_rels)):
            st.markdown(f"_Relation {i+1}_")
            rtype = st.text_input(f"Type {i+1}", key=f"r_type_{i}", value="RELATED_TO")
            target = st.selectbox(f"Target {i+1}", options=existing_names, key=f"r_tgt_{i}")
            rels.append((rtype, target))

        submitted = st.form_submit_button("Create Concept")
        if submitted:
            c = Concept(concept_name, description=concept_desc)
            # add features
            for fname, fmod, fvec in feats:
                vec = [float(x.strip()) for x in fvec.split(",") if x.strip() != ""]
                c.add_ensemble(FeatureEnsemble(fname, fmod, vec))
            # add relationships
            for rtype, tgt_name in rels:
                c.add_relationship(rtype, workspace[tgt_name].concept_id,
                                   description=f"user-defined to {tgt_name}")
            # add to workspace (handle name collision)
            base = c.name; suffix = 1; name = base
            while name in workspace:
                suffix += 1
                name = f"{base}_{suffix}"
            c.name = name
            workspace[name] = c
            st.success(f"Concept '{name}' created and added to workspace.")
