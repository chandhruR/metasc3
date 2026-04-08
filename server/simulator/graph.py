from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx


class GraphEngine:
    @staticmethod
    def neighborhood_edges(
        graph: nx.DiGraph,
        account_id: str,
        depth: int = 1,
        max_edges: int = 80,
    ) -> List[Tuple[str, str, Dict]]:
        seen: Set[Tuple[str, str]] = set()
        frontier = {account_id}
        results: List[Tuple[str, str, Dict]] = []

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node in frontier:
                for _u, v, data in graph.out_edges(node, data=True):
                    key = (node, v)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((node, v, dict(data)))
                    next_frontier.add(v)
                    if len(results) >= max_edges:
                        return results
                for u, _v, data in graph.in_edges(node, data=True):
                    key = (u, node)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append((u, node, dict(data)))
                    if len(results) >= max_edges:
                        return results
                    next_frontier.add(u)
            frontier = next_frontier
            if not frontier or len(results) >= max_edges:
                break
        return results

    @staticmethod
    def clusters_from_partition(
        graph: nx.DiGraph,
        account: str,
        cluster_attr: str = "cluster_id",
    ) -> Dict[str, List[str]]:
        by: Dict[str, List[str]] = defaultdict(list)
        for n in graph.nodes:
            by[str(graph.nodes[n].get(cluster_attr, "default"))].append(n)
        return dict(by)

    @staticmethod
    def bridge_accounts(graph: nx.DiGraph, cluster_attr: str = "cluster_id") -> List[str]:
        bridges: List[str] = []
        cluster_of = {n: str(graph.nodes[n].get(cluster_attr, "default")) for n in graph.nodes}
        for n in graph.nodes:
            neigh_clusters: Set[str] = set()
            for _u, v in graph.out_edges(n):
                neigh_clusters.add(cluster_of.get(v, "default"))
            for u, _v in graph.in_edges(n):
                neigh_clusters.add(cluster_of.get(u, "default"))
            home = cluster_of.get(n, "default")
            if len(neigh_clusters - {home}) >= 1:
                bridges.append(n)
        return bridges

    @staticmethod
    def remove_edge_if_exists(graph: nx.DiGraph, source: str, target: str) -> bool:
        if graph.has_edge(source, target):
            graph.remove_edge(source, target)
            return True
        return False

    @staticmethod
    def affected_clusters(post_reach: Dict[str, int], min_reach: int = 1) -> Set[str]:
        return {c for c, r in post_reach.items() if r >= min_reach}

    @staticmethod
    def subgraph_summary(graph: nx.DiGraph) -> Dict[str, Any]:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        }
