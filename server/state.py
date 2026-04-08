from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import networkx as nx

if TYPE_CHECKING:
    from server.models import ActionType, CascadeAction


@dataclass
class AccountNode:
    account_id: str
    is_bot: bool
    is_coordinated: bool
    is_real_user: bool
    influence_score: float
    radicalization_level: float
    content_type: str
    cluster_id: str
    status: str = "active"
    join_date: str = "2024-01-01"
    post_count: int = 10
    follower_count: int = 50
    following_count: int = 50
    engagement_rate: float = 0.05
    recent_topics: List[str] = field(default_factory=list)
    posting_frequency: float = 1.0
    flags: List[str] = field(default_factory=list)


@dataclass
class PostNode:
    post_id: str
    author_id: str
    is_harmful: bool
    harm_type: Optional[str]
    spread_count: int
    reach: int
    content_summary: str = ""
    timestamp: str = ""
    removed: bool = False
    downranked: bool = False
    has_friction: bool = False
    engagement: Dict[str, int] = field(default_factory=dict)
    cluster_reach: Dict[str, int] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)


@dataclass
class CommunityState:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    accounts: Dict[str, AccountNode] = field(default_factory=dict)
    posts: Dict[str, PostNode] = field(default_factory=dict)

    cascade_probability: float = 0.0
    tipping_point_reached: bool = False
    tipping_point_step: Optional[int] = None

    overall_health: float = 1.0
    cluster_temperatures: Dict[str, float] = field(default_factory=dict)

    origin_accounts: List[str] = field(default_factory=list)
    trigger_post: Optional[str] = None
    cascade_type: str = ""

    harassment_target_id: Optional[str] = None
    coordinated_accounts: Set[str] = field(default_factory=set)
    organic_harasser_accounts: Set[str] = field(default_factory=set)
    gateway_accounts: Set[str] = field(default_factory=set)

    removed_posts: Set[str] = field(default_factory=set)
    suspended_accounts: Set[str] = field(default_factory=set)
    shadow_banned_accounts: Set[str] = field(default_factory=set)
    removed_accounts: Set[str] = field(default_factory=set)
    friction_posts: Set[str] = field(default_factory=set)
    removed_edges: Set[Tuple[str, str]] = field(default_factory=set)

    actioned_real_users: Set[str] = field(default_factory=set)
    wrongly_removed_posts: Set[str] = field(default_factory=set)

    trending_topics: List[str] = field(default_factory=list)
    recent_flags: List[str] = field(default_factory=list)
    total_posts_today: int = 0

    prev_cascade_probability: float = 0.0
    prev_overall_health: float = 1.0

    intervention_count: int = 0
    radicalization_accelerated: bool = False

    _rng: Any = None

    def tick(self, step: int) -> None:
        from server.simulator.behavior import BehaviorSimulator
        from server.simulator.spread import SpreadSimulator

        self.prev_cascade_probability = self.cascade_probability
        self.prev_overall_health = self.overall_health

        SpreadSimulator.step_spread(self, self._rng, intensity=1.0)
        BehaviorSimulator.update_cluster_temperatures(self)
        BehaviorSimulator.organic_radicalization_tick(
            self, self._rng, gateway_accounts=self.gateway_accounts
        )

        self._update_cascade_probability(step)
        self._update_community_health()
        if self.tipping_point_step is not None and step >= self.tipping_point_step:
            self.tipping_point_reached = True

    def _update_cascade_probability(self, step: int) -> None:
        high_inf_rad = sum(
            1
            for a in self.accounts.values()
            if a.influence_score > 0.65 and a.radicalization_level > 0.55 and a.status == "active"
        )
        clusters_hot = sum(1 for t in self.cluster_temperatures.values() if t > 0.55)
        harmful_live = sum(1 for p in self.posts.values() if p.is_harmful and not p.removed)

        bridge_compromised = 0
        bridge_ids = []
        for n in self.graph.nodes:
            data = self.graph.nodes[n]
            cid = data.get("cluster_id", "")
            for _u, v in self.graph.out_edges(n):
                vd = self.graph.nodes.get(v, {})
                if vd.get("cluster_id", "") != cid:
                    bridge_ids.append(n)
                    break

        for b in bridge_ids[:50]:
            acc = self.accounts.get(b)
            if acc and acc.radicalization_level > 0.6:
                bridge_compromised += 1

        base = 0.05
        if self.cascade_type == "health_misinformation":
            base += harmful_live * 0.028
        elif self.cascade_type == "coordinated_harassment":
            coord_active = 0
            for c in self.coordinated_accounts:
                ac = self.accounts.get(c)
                if ac and ac.status == "active":
                    coord_active += 1
            base += min(0.45, coord_active * 0.02)
        elif self.cascade_type == "radicalization_pipeline":
            base += high_inf_rad * 0.015 + clusters_hot * 0.04

        tipping_pressure = 0.0
        if self.tipping_point_step:
            tipping_pressure = min(0.35, (step / max(self.tipping_point_step, 1)) * 0.25)

        self.cascade_probability = min(
            1.0,
            base + 0.04 * high_inf_rad + 0.03 * clusters_hot + 0.018 * bridge_compromised + tipping_pressure,
        )

    def _update_community_health(self) -> None:
        mean_r = (
            sum(a.radicalization_level for a in self.accounts.values()) / max(len(self.accounts), 1)
        )
        mean_temp = (
            sum(self.cluster_temperatures.values()) / max(len(self.cluster_temperatures), 1)
            if self.cluster_temperatures
            else 0.2
        )
        harmful_frac = (
            sum(1 for p in self.posts.values() if p.is_harmful and not p.removed)
            / max(len(self.posts), 1)
        )
        removal_shock = min(0.35, self.intervention_count * 0.008)

        self.overall_health = float(
            max(0.0, min(1.0, 1.0 - 0.55 * mean_r - 0.25 * mean_temp - 0.35 * harmful_frac - removal_shock))
        )

    def apply_intervention(self, action: "CascadeAction") -> Dict[str, Any]:
        from server.models import ActionType

        at = action.action_type
        result: Dict[str, Any] = {"action": at.value, "ok": True}

        if at == ActionType.OBSERVE:
            result["detail"] = "no_change"
            return result

        if at == ActionType.INSPECT_ACCOUNT:
            result["detail"] = "defer_to_env"
            return result
        if at == ActionType.INSPECT_POST:
            result["detail"] = "defer_to_env"
            return result
        if at == ActionType.INSPECT_NETWORK:
            result["detail"] = "defer_to_env"
            return result
        if at == ActionType.TRACE_SPREAD:
            result["detail"] = "defer_to_env"
            return result
        if at == ActionType.ANALYZE_CLUSTER:
            result["detail"] = "defer_to_env"
            return result
        if at == ActionType.SUBMIT_REPORT:
            result["detail"] = "report_submitted"
            return result

        self.intervention_count += 1

        if at == ActionType.REMOVE_POST and action.target_post_id:
            pid = action.target_post_id
            if pid in self.posts:
                self.posts[pid].removed = True
                self.removed_posts.add(pid)
                author = self.posts[pid].author_id
                acc = self.accounts.get(author)
                if acc and acc.is_real_user and not self.posts[pid].is_harmful:
                    self.wrongly_removed_posts.add(pid)
            result["detail"] = f"removed_post:{pid}"

        elif at == ActionType.DOWNRANK_POST and action.target_post_id:
            pid = action.target_post_id
            if pid in self.posts:
                self.posts[pid].downranked = True
            result["detail"] = f"downrank:{pid}"

        elif at == ActionType.ADD_FRICTION and action.target_post_id:
            pid = action.target_post_id
            if pid in self.posts:
                self.posts[pid].has_friction = True
                self.friction_posts.add(pid)
            result["detail"] = f"friction:{pid}"

        elif at == ActionType.SUSPEND_ACCOUNT and action.target_account_id:
            aid = action.target_account_id
            if aid in self.accounts:
                self.accounts[aid].status = "suspended"
                self.suspended_accounts.add(aid)
                if self.accounts[aid].is_real_user and aid not in self.coordinated_accounts and not self.accounts[aid].is_bot:
                    if self.cascade_type == "coordinated_harassment" and aid in self.organic_harasser_accounts:
                        self.actioned_real_users.add(aid)
                    elif self.cascade_type != "coordinated_harassment":
                        self.actioned_real_users.add(aid)
            result["detail"] = f"suspend:{aid}"

        elif at == ActionType.SHADOW_BAN_ACCOUNT and action.target_account_id:
            aid = action.target_account_id
            if aid in self.accounts:
                self.accounts[aid].status = "shadow_banned"
                self.shadow_banned_accounts.add(aid)
            result["detail"] = f"shadow:{aid}"

        elif at == ActionType.REMOVE_ACCOUNT and action.target_account_id:
            aid = action.target_account_id
            if aid in self.accounts:
                self.accounts[aid].status = "removed"
                self.removed_accounts.add(aid)
                if self.accounts[aid].is_real_user and aid not in self.coordinated_accounts:
                    self.actioned_real_users.add(aid)
            result["detail"] = f"remove_account:{aid}"

        elif at == ActionType.DISCONNECT_ACCOUNTS and action.target_edge:
            s, t = action.target_edge
            key = (s, t)
            if self.graph.has_edge(s, t):
                self.graph.remove_edge(s, t)
                self.removed_edges.add(key)
                if self.cascade_type == "radicalization_pipeline":
                    sa = self.accounts.get(s)
                    ta = self.accounts.get(t)
                    if sa and ta and sa.is_real_user and ta.is_real_user:
                        if s not in self.gateway_accounts and t not in self.gateway_accounts:
                            if sa.radicalization_level < 0.35 and ta.radicalization_level < 0.35:
                                self.radicalization_accelerated = True
            result["detail"] = f"disconnect:{s}->{t}"

        else:
            result["ok"] = False
            result["detail"] = "missing_target"

        fps = self.check_false_positives(action)
        if fps:
            result["false_positives"] = fps
        return result

    def check_false_positives(self, action: "CascadeAction") -> List[str]:
        from server.models import ActionType

        at = action.action_type
        fps: List[str] = []

        if at in (
            ActionType.SUSPEND_ACCOUNT,
            ActionType.SHADOW_BAN_ACCOUNT,
            ActionType.REMOVE_ACCOUNT,
        ) and action.target_account_id:
            aid = action.target_account_id
            acc = self.accounts.get(aid)
            if acc and acc.is_real_user and not acc.is_bot:
                if self.cascade_type == "coordinated_harassment":
                    if aid in self.organic_harasser_accounts or (
                        aid not in self.coordinated_accounts and aid != self.harassment_target_id
                    ):
                        if aid not in self.coordinated_accounts:
                            fps.append(aid)
                elif self.cascade_type == "health_misinformation":
                    if not acc.is_bot and aid not in self.origin_accounts:
                        if at == ActionType.REMOVE_ACCOUNT or at == ActionType.SUSPEND_ACCOUNT:
                            fps.append(aid)
                elif self.cascade_type == "radicalization_pipeline":
                    if aid in self.gateway_accounts:
                        pass
                    elif acc.radicalization_level < 0.35 and acc.content_type not in ("borderline",):
                        fps.append(aid)

        if at == ActionType.REMOVE_POST and action.target_post_id:
            pid = action.target_post_id
            p = self.posts.get(pid)
            if p and not p.is_harmful:
                fps.append(f"post:{pid}")

        return fps
