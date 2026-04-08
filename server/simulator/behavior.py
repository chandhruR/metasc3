from __future__ import annotations

from typing import Dict

import networkx as nx

from server.state import CommunityState


class BehaviorSimulator:
    @staticmethod
    def update_cluster_temperatures(state: CommunityState) -> None:
        graph = state.graph
        cluster_rad: Dict[str, list] = {}
        cluster_count: Dict[str, int] = {}

        for aid, acc in state.accounts.items():
            cid = acc.cluster_id
            cluster_rad.setdefault(cid, []).append(acc.radicalization_level)
            cluster_count[cid] = cluster_count.get(cid, 0) + 1

        for cid, rads in cluster_rad.items():
            mean_r = sum(rads) / max(len(rads), 1)
            harmed_posts = sum(
                1 for p in state.posts.values() if p.is_harmful and not p.removed
            )
            community_factor = min(1.0, harmed_posts / 25.0)
            state.cluster_temperatures[cid] = min(1.0, 0.15 + 0.7 * mean_r + 0.25 * community_factor)

    @staticmethod
    def organic_radicalization_tick(state: CommunityState, rng, gateway_accounts: set | None = None) -> None:
        gateway_accounts = gateway_accounts or set()
        for aid, acc in state.accounts.items():
            if acc.status != "active":
                continue
            bump = 0.0
            if aid in gateway_accounts:
                bump += 0.003
            if acc.is_coordinated:
                bump += 0.002
            if acc.content_type == "borderline":
                bump += 0.004
            if rng.random() < 0.05:
                acc.radicalization_level = min(1.0, acc.radicalization_level + bump)
