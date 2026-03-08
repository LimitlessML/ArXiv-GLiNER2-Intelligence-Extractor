import streamlit as st
import pandas as pd
from extractor.fetcher import fetch_paper
from extractor.ner import load_model, extract_from_paper, ENTITY_LABELS

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
            sections = ["All sections"] + list(all_results.keys())
            selected_section = st.selectbox("Filter by section", sections)

            filtered = all_entities if selected_section == "All sections" else [
                e for e in all_entities if e["section"] == selected_section
            ]

            df = pd.DataFrame(filtered)
            cols = st.columns(len(ENTITY_LABELS))
            for i, label in enumerate(ENTITY_LABELS):
                subset = df[df["label"] == label]["text"].tolist()
                with cols[i]:
                    st.markdown(f"**{label}**")
                    for item in subset:
                        st.markdown(f"- {item}")

            st.divider()
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "entities.csv", "text/csv")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
