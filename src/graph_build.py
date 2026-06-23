from __future__ import annotations

import pandas as pd
import networkx as nx


def build_di_graph(entities: pd.DataFrame, relationships: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()

    for _, row in entities.iterrows():
        lei = row.get("lei")
        if pd.isna(lei):
            continue
        g.add_node(
            lei,
            legal_name=row.get("legal_name"),
            country=row.get("country_legal"),
            city=row.get("city_legal"),
            category=row.get("category"),
            legal_form=row.get("legal_form"),
        )

    for _, row in relationships.iterrows():
        src = row.get("source_lei")
        dst = row.get("target_lei")
        if pd.notna(src) and pd.notna(dst):
            g.add_edge(
                src,
                dst,
                relationship_type=row.get("relationship_type"),
                relationship_status=row.get("relationship_status"),
                accounting_standard=row.get("accounting_standard"),
                period_end=row.get("period_end"),
            )

    return g


def graph_summary(g: nx.DiGraph) -> pd.DataFrame:
    columns = [
        "lei",
        "legal_name",
        "country",
        "city",
        "category",
        "legal_form",
        "in_degree",
        "out_degree",
        "weakly_connected_component_size",
    ]
    if len(g) == 0:
        return pd.DataFrame(columns=columns)

    component_sizes = {}
    for component in nx.connected_components(g.to_undirected()):
        size = len(component)
        for node in component:
            component_sizes[node] = size

    rows = []
    for node, attrs in g.nodes(data=True):
        rows.append({
            "lei": node,
            "legal_name": attrs.get("legal_name"),
            "country": attrs.get("country"),
            "city": attrs.get("city"),
            "category": attrs.get("category"),
            "legal_form": attrs.get("legal_form"),
            "in_degree": g.in_degree(node),
            "out_degree": g.out_degree(node),
            "weakly_connected_component_size": component_sizes.get(node, 1),
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["weakly_connected_component_size", "in_degree", "out_degree"],
        ascending=False,
    )


def parent_flags(relationships: pd.DataFrame) -> pd.DataFrame:
    if relationships.empty:
        return pd.DataFrame(columns=["lei", "has_parent_link", "has_ultimate_link"])

    direct = (
        relationships.loc[relationships["relationship_type"] == "direct_parent", ["source_lei"]]
        .drop_duplicates()
        .assign(has_parent_link=1)
        .rename(columns={"source_lei": "lei"})
    )

    ultimate = (
        relationships.loc[relationships["relationship_type"] == "ultimate_parent", ["source_lei"]]
        .drop_duplicates()
        .assign(has_ultimate_link=1)
        .rename(columns={"source_lei": "lei"})
    )

    out = direct.merge(ultimate, on="lei", how="outer").fillna(0)
    out["has_parent_link"] = out["has_parent_link"].astype(int)
    out["has_ultimate_link"] = out["has_ultimate_link"].astype(int)
    return out


def add_graph_features(g: nx.DiGraph, summary: pd.DataFrame) -> pd.DataFrame:
    if len(g) == 0:
        out = summary.copy()
        out["pagerank"] = 0.0
        out["degree_centrality"] = 0.0
        return out

    pagerank = nx.pagerank(g)
    ug = g.to_undirected()
    degree_centrality = nx.degree_centrality(ug)

    out = summary.copy()
    out["pagerank"] = out["lei"].map(pagerank).fillna(0.0)
    out["degree_centrality"] = out["lei"].map(degree_centrality).fillna(0.0)
    return out
