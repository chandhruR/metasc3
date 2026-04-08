from __future__ import annotations

import random
from datetime import date, timedelta
from typing import List, Tuple

import networkx as nx

from server.state import AccountNode, CommunityState, PostNode


def build_parenting_misinfo_community(seed: int, n_accounts: int = 500) -> CommunityState:
    rng = random.Random(seed)
    state = CommunityState(_rng=rng)
    state.cascade_type = "health_misinformation"
    state.tipping_point_step = 12
    state.trending_topics = ["parenting", "sleep_training", "vaccines", "pediatrics"]
    state.recent_flags = ["uptick_reports_health", "new_member_spike"]
    state.total_posts_today = rng.randint(40, 120)

    clusters = [
        ("new_parents", max(2, n_accounts * 45 // 100)),
        ("experienced_parents", max(2, n_accounts * 40 // 100)),
        ("skeptics", max(2, n_accounts - (n_accounts * 45 // 100) - (n_accounts * 40 // 100))),
    ]

    accounts: List[str] = []
    cluster_of: dict[str, str] = {}
    for cid, cnt in clusters:
        for _ in range(cnt):
            aid = f"{cid}_{len(accounts)}"
            accounts.append(aid)
            cluster_of[aid] = cid

    origin_id = "origin_misinfo_bot"
    amplifier_ids = ["amp_parent_a", "amp_parent_b", "amp_parent_c"]
    leader_id = "community_leader_elena"
    trigger_post = "post_false_vaccine_claim"

    all_ids = accounts + [origin_id] + amplifier_ids + [leader_id]
    state.graph.add_nodes_from(all_ids)

    topics_by = {
        "new_parents": ["sleep", "feeding", "vaccines", "milestones"],
        "experienced_parents": ["school", "tantrums", "schedules"],
        "skeptics": ["alternative_health", "mainstream_critique", "organic"],
    }

    def make_account(
        aid: str, bot: bool, coord: bool, real: bool, influence: float, cluster: str, **kw
    ) -> AccountNode:
        return AccountNode(
            account_id=aid,
            is_bot=bot,
            is_coordinated=coord,
            is_real_user=real,
            influence_score=influence,
            radicalization_level=kw.get("rad", 0.05),
            content_type=kw.get("content_type", "normal"),
            cluster_id=cluster,
            join_date=str(date.today() - timedelta(days=kw.get("joined_days_ago", rng.randint(30, 900)))),
            post_count=kw.get("post_count", rng.randint(5, 200)),
            follower_count=kw.get("followers", rng.randint(20, 400)),
            following_count=kw.get("following", rng.randint(20, 350)),
            engagement_rate=kw.get("engagement", rng.uniform(0.02, 0.12)),
            recent_topics=list(rng.sample(topics_by[cluster], min(3, len(topics_by[cluster])))),
            posting_frequency=kw.get("freq", rng.uniform(0.5, 3.0)),
            flags=kw.get("flags", []),
        )

    for aid in accounts:
        cid = cluster_of[aid]
        inf = rng.uniform(0.2, 0.75)
        st = make_account(aid, False, False, True, inf, cid)
        state.accounts[aid] = st
        state.graph.nodes[aid]["cluster_id"] = cid

    state.accounts[origin_id] = make_account(
        origin_id,
        True,
        False,
        False,
        0.25,
        "skeptics",
        joined_days_ago=4,
        followers=12,
        following=38,
        post_count=3,
        flags=["very_new_account", "low_trust_score", "health_topic_spike"],
        engagement=0.01,
    )
    state.graph.nodes[origin_id]["cluster_id"] = "skeptics"

    for amp in amplifier_ids:
        st = make_account(amp, False, False, True, rng.uniform(0.35, 0.55), "new_parents", rad=0.15)
        st.flags = ["shared_health_content_recently", "high_empathy_signals"]
        state.accounts[amp] = st
        state.graph.nodes[amp]["cluster_id"] = "new_parents"

    state.accounts[leader_id] = make_account(
        leader_id,
        False,
        False,
        True,
        0.92,
        "new_parents",
        followers=18000,
        post_count=840,
        engagement=0.11,
        flags=["community_leader", "verified_adjacent"],
    )
    state.graph.nodes[leader_id]["cluster_id"] = "new_parents"

    def add_edge(u: str, v: str, rel: str, strength: float):
        if not state.graph.has_edge(u, v):
            state.graph.add_edge(u, v, relationship=rel, strength=strength, created_at="2024-03-01")

    for aid in accounts:
        k = rng.randint(1, 4)
        others = [x for x in accounts if cluster_of[x] == cluster_of[aid] and x != aid]
        rng.shuffle(others)
        for peer in others[:k]:
            if rng.random() < 0.45:
                add_edge(aid, peer, "interacted", rng.uniform(0.2, 0.8))

    bridge_pairs: List[Tuple[str, str]] = []
    for _ in range(max(8, n_accounts // 60)):
        a = rng.choice([x for x in accounts if cluster_of[x] == "new_parents"])
        b = rng.choice([x for x in accounts if cluster_of[x] == "skeptics"])
        bridge_pairs.append((a, b))
    for u, v in bridge_pairs[:20]:
        add_edge(u, v, "interacted", rng.uniform(0.15, 0.55))

    path_edges = [
        (origin_id, amplifier_ids[0], 0.55),
        (amplifier_ids[0], amplifier_ids[1], 0.62),
        (amplifier_ids[1], amplifier_ids[2], 0.58),
        (amplifier_ids[2], leader_id, 0.66),
    ]
    for u, v, s in path_edges:
        add_edge(u, v, "interacted", s)

    for amp in amplifier_ids:
        add_edge(amp, leader_id, "follower", rng.uniform(0.25, 0.45))

    state.origin_accounts = [origin_id]
    state.trigger_post = trigger_post

    engagement = {"likes": 3, "shares": 1, "comments": 6, "reports": 4}
    state.posts[trigger_post] = PostNode(
        post_id=trigger_post,
        author_id=origin_id,
        is_harmful=True,
        harm_type="health_misinformation",
        spread_count=2,
        reach=240,
        content_summary="Claims routine childhood vaccine batch caused seizures; urges alternate schedule.",
        timestamp="2024-04-02T09:15:00Z",
        engagement=dict(engagement),
        cluster_reach={"skeptics": 120, "new_parents": 80},
        flags=["user_reports_health", "velocity_anomaly"],
    )

    state.cluster_temperatures = {c[0]: rng.uniform(0.2, 0.45) for c in clusters}
    state.overall_health = 0.78
    state.cascade_probability = 0.08
    return state
