from __future__ import annotations

import random

from server.state import AccountNode, CommunityState, PostNode


def build_radicalization_pipeline_community(seed: int, n_accounts: int = 5000) -> CommunityState:
    rng = random.Random(seed)
    state = CommunityState(_rng=rng)
    state.cascade_type = "radicalization_pipeline"
    state.tipping_point_step = 35
    state.trending_topics = ["hobby_sport", "history_deepdives", "current_events", "philosophy"]
    state.recent_flags = ["gradual_topic_shift", "cross_cluster_quote_chains"]
    state.total_posts_today = rng.randint(800, 2500)

    cluster_names = ["casual_hobby", "political_curious", "ideology_adjacent", "high_intensity"]
    cluster_sizes = [
        max(2, int(n_accounts * 0.38)),
        max(2, int(n_accounts * 0.28)),
        max(2, int(n_accounts * 0.22)),
        max(2, n_accounts - sum([max(2, int(n_accounts * 0.38)), max(2, int(n_accounts * 0.28)), max(2, int(n_accounts * 0.22))])),
    ]

    accounts: list[str] = []
    cluster_of: dict[str, str] = {}
    for cname, sz in zip(cluster_names, cluster_sizes):
        for _ in range(sz):
            aid = f"{cname}_{len(accounts)}"
            accounts.append(aid)
            cluster_of[aid] = cname

    gateway_accounts = [f"gateway_{i}" for i in range(8)]
    state.graph.add_nodes_from(accounts + gateway_accounts)

    bridge_pairs = []
    for i in range(len(cluster_names) - 1):
        left = [a for a in accounts if cluster_of[a] == cluster_names[i]]
        right = [a for a in accounts if cluster_of[a] == cluster_names[i + 1]]
        for g in gateway_accounts[i * 2 : i * 2 + 2]:
            bridge_pairs.append((g, rng.choice(left)))
            bridge_pairs.append((g, rng.choice(right)))

    for aid in accounts:
        cname = cluster_of[aid]
        temp_topic = rng.choice(state.trending_topics)
        rad_base = {"casual_hobby": 0.08, "political_curious": 0.22, "ideology_adjacent": 0.42, "high_intensity": 0.66}
        state.accounts[aid] = AccountNode(
            account_id=aid,
            is_bot=False,
            is_coordinated=False,
            is_real_user=True,
            influence_score=rng.uniform(0.15, 0.75),
            radicalization_level=rad_base[cname] + rng.uniform(-0.05, 0.06),
            content_type="normal",
            cluster_id=cname,
            follower_count=rng.randint(30, 6000),
            flags=[],
            recent_topics=[temp_topic],
        )
        state.graph.nodes[aid]["cluster_id"] = cname

    for g in gateway_accounts:
        state.gateway_accounts.add(g)
        state.accounts[g] = AccountNode(
            account_id=g,
            is_bot=False,
            is_coordinated=False,
            is_real_user=True,
            influence_score=rng.uniform(0.45, 0.8),
            radicalization_level=rng.uniform(0.25, 0.48),
            content_type="borderline",
            cluster_id="bridge",
            follower_count=rng.randint(800, 12000),
            flags=["cross_posts_multiple_clusters", "incremental_extremity"],
            recent_topics=["philosophy", "current_events"],
        )
        state.graph.nodes[g]["cluster_id"] = "bridge"

    def add_edge(u: str, v: str, s: float):
        state.graph.add_edge(u, v, relationship="interacted", strength=s, created_at="2024-02-10")

    for aid in accounts:
        peers = [x for x in accounts if cluster_of[x] == cluster_of[aid] and x != aid]
        rng.shuffle(peers)
        for p in peers[: rng.randint(1, 3)]:
            if rng.random() < 0.35:
                add_edge(aid, p, rng.uniform(0.15, 0.55))

    for u, v in bridge_pairs:
        add_edge(u, v, rng.uniform(0.35, 0.85))

    narrative_post = "post_gradient_narrative_snippet"
    state.trigger_post = narrative_post
    state.posts[narrative_post] = PostNode(
        post_id=narrative_post,
        author_id=gateway_accounts[0],
        is_harmful=False,
        harm_type=None,
        spread_count=12,
        reach=9000,
        content_summary="Incremental framing shift: distrust of institutions framed as 'just asking questions'.",
        timestamp="2024-04-11T18:40:00Z",
        engagement={"likes": 450, "shares": 180, "comments": 120, "reports": 2},
        flags=["borderline_policy", "cross_cluster_traction_mild"],
    )

    state.origin_accounts = list(gateway_accounts)
    state.cluster_temperatures = {c: 0.25 + i * 0.12 for i, c in enumerate(cluster_names)}
    state.cluster_temperatures["bridge"] = 0.48
    state.overall_health = 0.63
    state.cascade_probability = 0.12
    return state
