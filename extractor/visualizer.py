import itertools
import networkx as nx
import plotly.graph_objects as go

LABEL_COLORS = {
    "ML model":           "#4e79a7",
    "dataset":            "#f28e2b",
    "metric":             "#e15759",
    "institution":        "#76b7b2",
    "method or technique":"#59a14f",
    "benchmark":          "#edc948",
    "author":             "#b07aa1",
}


def build_knowledge_graph(all_results: dict) -> go.Figure:
    # Count co-occurrences between entities across sections
    cooccurrence = {}
    node_info = {}

    for _, entities in all_results.items():
        for e in entities:
            nid = f"{e['text']}|{e['label']}"
            node_info[nid] = e
        pairs = [f"{e['text']}|{e['label']}" for e in entities]
        for a, b in itertools.combinations(pairs, 2):
            key = tuple(sorted([a, b]))
            cooccurrence[key] = cooccurrence.get(key, 0) + 1

    # Only keep edges with co-occurrence >= 1, limit to top 40 edges
    edges = sorted(cooccurrence.items(), key=lambda x: -x[1])[:40]
    active_nodes = set()
    for (a, b), _ in edges:
        active_nodes.add(a)
        active_nodes.add(b)

    # Build networkx graph for layout
    G = nx.Graph()
    G.add_nodes_from(active_nodes)
    for (a, b), w in edges:
        G.add_edge(a, b, weight=w)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Build plotly traces
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#444444"),
        hoverinfo="none",
    )

    node_traces = []
    for label, color in LABEL_COLORS.items():
        nodes_in_group = [n for n in active_nodes if node_info[n]["label"] == label]
        if not nodes_in_group:
            continue
        nx_ = [pos[n][0] for n in nodes_in_group]
        ny_ = [pos[n][1] for n in nodes_in_group]
        texts = [node_info[n]["text"] for n in nodes_in_group]
        node_traces.append(go.Scatter(
            x=nx_, y=ny_,
            mode="markers+text",
            name=label,
            text=texts,
            textposition="top center",
            textfont=dict(size=9, color="white"),
            marker=dict(size=14, color=color, line=dict(width=1, color="#222222")),
            hovertemplate="%{text}<br>" + label + "<extra></extra>",
        ))

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(bgcolor="#1a1a2e", bordercolor="#333", borderwidth=1),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=20, b=20),
            height=550,
        )
    )
    return fig


def build_radar_chart(all_results: dict) -> go.Figure:
    counts = {label: 0 for label in LABEL_COLORS}
    for entities in all_results.values():
        for e in entities:
            if e["label"] in counts:
                counts[e["label"]] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(78, 121, 167, 0.3)",
        line=dict(color="#4e79a7", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, color="#888888")),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_summary(all_results: dict, metadata: dict) -> str:
    def get(label, section_hint=None):
        for section, entities in all_results.items():
            if section_hint and section_hint not in section:
                continue
            for e in entities:
                if e["label"] == label:
                    return e["text"]
        return None

    models = [e["text"] for s in all_results.values() for e in s if e["label"] == "ML model"][:3]
    datasets = [e["text"] for s in all_results.values() for e in s if e["label"] == "dataset"][:3]
    methods = [e["text"] for s in all_results.values() for e in s if e["label"] == "method or technique"][:2]
    metrics = [e["text"] for s in all_results.values() for e in s if e["label"] == "metric"][:2]
    institutions = [e["text"] for s in all_results.values() for e in s if e["label"] == "institution"][:2]

    parts = []
    if models:
        parts.append(f"introduces **{', '.join(models)}**")
    if methods:
        parts.append(f"using **{', '.join(methods)}**")
    if institutions:
        parts.append(f"from **{', '.join(institutions)}**")
    if datasets:
        parts.append(f"evaluated on **{', '.join(datasets)}**")
    if metrics:
        parts.append(f"measuring **{', '.join(metrics)}**")

    if not parts:
        return "No structured summary could be generated."
    return "This paper " + ", ".join(parts) + "."
