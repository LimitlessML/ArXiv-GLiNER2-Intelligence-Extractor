import streamlit as st
import pandas as pd
from extractor.fetcher import fetch_paper
from extractor.ner import load_model, extract_from_paper, ENTITY_LABELS
from extractor.visualizer import build_knowledge_graph, build_radar_chart, build_summary
from extractor.linker import link_entities

st.set_page_config(
    page_title="ArXiv Intelligence Extractor",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource
def get_model():
    return load_model()

st.title("ArXiv Paper Intelligence Extractor")
st.caption("Zero-shot NER on any ArXiv paper using GLiNER2")

with st.spinner("Loading GLiNER2 model..."):
    model = get_model()

mode = st.radio("Mode", ["Single paper", "Compare two papers"], horizontal=True)

# ─────────────────────────────────────────────────────────────
# SINGLE PAPER MODE
# ─────────────────────────────────────────────────────────────
if mode == "Single paper":
    input_str = st.text_input(
        "ArXiv ID or URL",
        placeholder="e.g. 2507.18546 or https://arxiv.org/abs/2507.18546",
    )

    if st.button("Extract", type="primary") and input_str:
        try:
            with st.spinner("Fetching paper..."):
                paper = fetch_paper(input_str)

            st.subheader(paper["title"])
            st.caption(f"Authors: {', '.join(paper['authors'][:5])}")
            st.caption(f"Published: {paper['published']} | Source: {paper['source'].upper()}")

            with st.expander("Abstract"):
                st.write(paper["abstract"])

            with st.spinner("Extracting entities..."):
                all_results = extract_from_paper(model, paper)

            all_entities = []
            for section, entities in all_results.items():
                for e in entities:
                    all_entities.append({**e, "section": section})

            if not all_entities:
                st.warning("No entities found.")
            else:
                st.markdown("### Intelligence Summary")
                st.info(build_summary(all_results, paper))

                col1, col2, col3 = st.columns(3)
                col1.metric("Entities found", len(all_entities))
                col2.metric("Sections analyzed", len(all_results))
                col3.metric("Entity types", len(ENTITY_LABELS))

                st.divider()

                tab1, tab2, tab3, tab4 = st.tabs([
                    "Entity Table", "Knowledge Graph", "Paper Fingerprint", "Raw Data"
                ])

                with tab1:
                    sections = ["All sections"] + list(all_results.keys())
                    selected_section = st.selectbox("Filter by section", sections)
                    filtered = all_entities if selected_section == "All sections" else [
                        e for e in all_entities if e["section"] == selected_section
                    ]
                    df_filtered = pd.DataFrame(filtered)
                    cols = st.columns(len(ENTITY_LABELS))
                    for i, label in enumerate(ENTITY_LABELS):
                        subset = df_filtered[df_filtered["label"] == label]["text"].tolist()
                        with cols[i]:
                            st.markdown(f"**{label}**")
                            for item in subset:
                                st.markdown(f"- {item}")

                with tab2:
                    st.markdown("**Entity co-occurrence** — colored by type, edges = shared section")
                    st.plotly_chart(build_knowledge_graph(all_results), use_container_width=True)

                with tab3:
                    st.markdown("**Paper fingerprint** — entity distribution by category")
                    st.plotly_chart(build_radar_chart(all_results), use_container_width=True)

                with tab4:
                    with st.spinner("Linking entities to Papers With Code..."):
                        linked = link_entities(all_entities)
                    df = pd.DataFrame(linked)
                    df["pwc_url"] = df["pwc_url"].fillna("")
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "pwc_url": st.column_config.LinkColumn("Papers With Code")
                        },
                    )
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, "entities.csv", "text/csv")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ─────────────────────────────────────────────────────────────
# COMPARE TWO PAPERS MODE
# ─────────────────────────────────────────────────────────────
else:
    col_a, col_b = st.columns(2)
    with col_a:
        input_a = st.text_input("Paper A", placeholder="e.g. 2507.18546")
    with col_b:
        input_b = st.text_input("Paper B", placeholder="e.g. 2403.09611")

    if st.button("Compare", type="primary") and input_a and input_b:
        try:
            with st.spinner("Fetching papers..."):
                paper_a = fetch_paper(input_a)
                paper_b = fetch_paper(input_b)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**A:** {paper_a['title'][:80]}...")
            with col_b:
                st.markdown(f"**B:** {paper_b['title'][:80]}...")

            with st.spinner("Extracting entities from both papers..."):
                results_a = extract_from_paper(model, paper_a)
                results_b = extract_from_paper(model, paper_b)

            flat_a = {(e["text"].lower(), e["label"]): e["text"]
                      for s in results_a.values() for e in s}
            flat_b = {(e["text"].lower(), e["label"]): e["text"]
                      for s in results_b.values() for e in s}

            shared_keys  = flat_a.keys() & flat_b.keys()
            only_a_keys  = flat_a.keys() - flat_b.keys()
            only_b_keys  = flat_b.keys() - flat_a.keys()

            col1, col2, col3 = st.columns(3)
            col1.metric("Shared entities", len(shared_keys))
            col2.metric(f"Only in A", len(only_a_keys))
            col3.metric(f"Only in B", len(only_b_keys))

            st.divider()

            tab_shared, tab_a, tab_b, tab_graphs = st.tabs([
                "Shared", "Only in A", "Only in B", "Fingerprints"
            ])

            def render_entity_grid(keys, flat):
                rows = [{"text": flat[k], "label": k[1]} for k in keys]
                if not rows:
                    st.info("None.")
                    return
                df = pd.DataFrame(rows)
                cols = st.columns(len(ENTITY_LABELS))
                for i, label in enumerate(ENTITY_LABELS):
                    subset = df[df["label"] == label]["text"].tolist()
                    with cols[i]:
                        st.markdown(f"**{label}**")
                        for item in subset:
                            st.markdown(f"- {item}")

            with tab_shared:
                render_entity_grid(shared_keys, flat_a)

            with tab_a:
                render_entity_grid(only_a_keys, flat_a)

            with tab_b:
                render_entity_grid(only_b_keys, flat_b)

            with tab_graphs:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Paper A fingerprint**")
                    st.plotly_chart(build_radar_chart(results_a), use_container_width=True)
                with c2:
                    st.markdown(f"**Paper B fingerprint**")
                    st.plotly_chart(build_radar_chart(results_b), use_container_width=True)

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
