from __future__ import annotations

from typing import Dict, List

from server.state import CommunityState


class SpreadSimulator:
    @staticmethod
    def step_spread(state: CommunityState, rng, intensity: float = 1.0) -> None:
        graph = state.graph
        for pid, post in list(state.posts.items()):
            if post.removed:
                continue
            if not post.is_harmful:
                continue
            author = post.author_id
            if author not in graph:
                continue
            visibility = 1.0
            if post.downranked:
                visibility *= 0.35
            if post.has_friction:
                visibility *= 0.6

            neighbors: List[str] = []
            for _u, v in graph.out_edges(author):
                acc = state.accounts.get(v)
                if acc and acc.status == "active":
                    neighbors.append(v)
            for u, _v in graph.in_edges(author):
                acc = state.accounts.get(u)
                if acc and acc.status == "active":
                    neighbors.append(u)

            if not neighbors:
                continue

            spread_budget = max(1, int(2 * visibility * intensity))
            rng.shuffle(neighbors)
            for target in neighbors[:spread_budget]:
                edge_data = {}
                if graph.has_edge(author, target):
                    edge_data = graph[author][target]
                elif graph.has_edge(target, author):
                    edge_data = graph[target][author]
                strength = float(edge_data.get("strength", 0.5))
                p = 0.12 * strength * visibility * intensity
                if rng.random() < p:
                    post.spread_count += 1
                    post.reach += state.accounts[target].follower_count // 50 + 1
                    state.accounts[target].radicalization_level = min(
                        1.0,
                        state.accounts[target].radicalization_level + 0.02 * strength,
                    )

    @staticmethod
    def trace_from_post(state: CommunityState, post_id: str, max_hops: int = 4) -> Dict:
        post = state.posts.get(post_id)
        if not post:
            return {"error": "unknown_post"}

        graph = state.graph
        visited = {post.author_id}
        frontier = [post.author_id]
        hops = 0
        layers = []

        while frontier and hops < max_hops:
            layer = []
            nxt = []
            for node in frontier:
                for _u, v in graph.out_edges(node):
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
                        layer.append({"from": node, "to": v})
                for u, _v in graph.in_edges(node):
                    if u not in visited:
                        visited.add(u)
                        nxt.append(u)
                        layer.append({"from": u, "to": node})
            if layer:
                layers.append(layer)
            frontier = nxt
            hops += 1

        return {
            "post_id": post_id,
            "author_id": post.author_id,
            "paths_layers": layers,
            "accounts_touched": list(visited),
        }
