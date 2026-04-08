from __future__ import annotations

import random
from datetime import datetime, timedelta

from server.state import AccountNode, CommunityState, PostNode


def build_coordinated_harassment_community(seed: int, n_accounts: int = 2000) -> CommunityState:
    rng = random.Random(seed)
    state = CommunityState(_rng=rng)
    state.cascade_type = "coordinated_harassment"
    state.tipping_point_step = 20
    state.trending_topics = ["policy_debate", "local_news", "creator_drama"]
    state.recent_flags = ["burst_reporting_pattern", "reply_chain_velocity_high"]
    state.total_posts_today = rng.randint(300, 900)

    target_id = "target_creator_alex"
    base_accounts = [f"user_{i}" for i in range(n_accounts)]
    coordinated = [f"coord_{i}" for i in range(15)]
    organic = [f"organic_piler_{i}" for i in range(40)]

    state.harassment_target_id = target_id
    state.graph.add_nodes_from(base_accounts + coordinated + organic + [target_id])

    def base_user(aid: str) -> AccountNode:
        return AccountNode(
            account_id=aid,
            is_bot=False,
            is_coordinated=False,
            is_real_user=True,
            influence_score=rng.uniform(0.1, 0.55),
            radicalization_level=rng.uniform(0.0, 0.25),
            content_type="normal",
            cluster_id="general",
            follower_count=rng.randint(50, 2000),
            following_count=rng.randint(100, 1500),
            recent_topics=list(rng.sample(state.trending_topics, min(2, len(state.trending_topics)))),
            flags=[],
        )

    for aid in base_accounts:
        state.accounts[aid] = base_user(aid)
        state.graph.nodes[aid]["cluster_id"] = "general"

    burst_start = datetime(2024, 5, 1, 14, 0, 0)
    for i, aid in enumerate(coordinated):
        join_offset = timedelta(seconds=i * 7)
        flags = ["tight_posting_window", "similar_language_template", "fresh_account_cluster"]
        state.accounts[aid] = AccountNode(
            account_id=aid,
            is_bot=True,
            is_coordinated=True,
            is_real_user=False,
            influence_score=rng.uniform(0.2, 0.4),
            radicalization_level=0.4,
            content_type="harassment",
            cluster_id="coord_blob",
            join_date=(burst_start + join_offset).date().isoformat(),
            post_count=rng.randint(8, 40),
            follower_count=rng.randint(5, 120),
            flags=flags,
            recent_topics=["pile_on", "target_creator_alex"],
            posting_frequency=rng.uniform(8.0, 20.0),
        )
        state.coordinated_accounts.add(aid)
        state.graph.nodes[aid]["cluster_id"] = "coord_blob"

    for i, aid in enumerate(organic):
        state.accounts[aid] = AccountNode(
            account_id=aid,
            is_bot=False,
            is_coordinated=False,
            is_real_user=True,
            influence_score=rng.uniform(0.25, 0.7),
            radicalization_level=rng.uniform(0.05, 0.35),
            content_type="commentary",
            cluster_id="general",
            follower_count=rng.randint(120, 4000),
            recent_topics=["creator_drama", "policy_debate"],
            flags=["organic_engagement", "heterogeneous_timing"],
        )
        state.organic_harasser_accounts.add(aid)
        state.graph.nodes[aid]["cluster_id"] = "general"

    state.accounts[target_id] = AccountNode(
        account_id=target_id,
        is_bot=False,
        is_coordinated=False,
        is_real_user=True,
        influence_score=0.88,
        radicalization_level=0.1,
        content_type="creator",
        cluster_id="creator",
        follower_count=250_000,
        post_count=1200,
        flags=["pile_on_target", "elevated_mentions"],
        recent_topics=["mental_health", "policy_debate"],
    )
    state.graph.nodes[target_id]["cluster_id"] = "creator"

    def add_edge(u: str, v: str, strength: float):
        state.graph.add_edge(u, v, relationship="interacted", strength=strength, created_at="2024-05-01")

    for c in coordinated:
        add_edge(c, target_id, rng.uniform(0.55, 0.95))
    for o in organic:
        if rng.random() < 0.65:
            add_edge(o, target_id, rng.uniform(0.2, 0.6))
        peer = rng.choice(coordinated)
        if rng.random() < 0.35:
            add_edge(o, peer, rng.uniform(0.1, 0.35))

    for _ in range(n_accounts // 8):
        u, v = rng.sample(base_accounts, 2)
        if not state.graph.has_edge(u, v) and rng.random() < 0.12:
            add_edge(u, v, rng.uniform(0.05, 0.3))

    trigger = "post_target_takedown_rumor"
    state.trigger_post = trigger
    state.posts[trigger] = PostNode(
        post_id=trigger,
        author_id=coordinated[0],
        is_harmful=True,
        harm_type="harassment_organizing",
        spread_count=35,
        reach=50_000,
        content_summary="Rumor alleging fabricated scandal; mobilizes pile-on against target creator.",
        timestamp="2024-05-01T14:00:15Z",
        engagement={"likes": 900, "shares": 400, "comments": 2200, "reports": 60},
        flags=["coordination_suspected"],
    )
    state.origin_accounts = list(coordinated)

    state.cluster_temperatures = {"general": 0.55, "coord_blob": 0.82, "creator": 0.72}
    state.overall_health = 0.52
    state.cascade_probability = 0.18
    return state
