from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from server.graders import Task1Grader, Task2Grader, Task3Grader
from server.graders.base import CascadeGrader
from server.models import (
    AccountSnapshot,
    AccountStatus,
    CascadeAction,
    CascadeReward,
    CommunityCluster,
    CommunityObservation,
    NetworkEdge,
    PostSnapshot,
    ResetResult,
    RiskLevel,
    StateResult,
    StepResult,
    ActionType,
)
from server.scenarios.generator import CommunityGenerator
from server.simulator.graph import GraphEngine
from server.simulator.spread import SpreadSimulator
from server.state import CommunityState

TASK_MAX_STEPS = {
    "task1_health_misinfo": 20,
    "task2_coordinated_harassment": 30,
    "task3_radicalization_pipeline": 50,
}


class CascadeEnvironment:
    def __init__(self) -> None:
        self.generator = CommunityGenerator()
        self.community: Optional[CommunityState] = None
        self.task_id: str = "task1_health_misinfo"
        self.step_idx: int = 0
        self.max_steps: int = 20
        self.initialized: bool = False
        self.cumulative_reward: float = 0.0
        self.actions_taken: List[str] = []
        self.interventions_made: List[Dict[str, Any]] = []
        self.grader: CascadeGrader = Task1Grader()
        self._grader_scores: Dict[str, float] = {}

    def _select_grader(self, task_id: str) -> CascadeGrader:
        if task_id == "task1_health_misinfo":
            return Task1Grader()
        if task_id == "task2_coordinated_harassment":
            return Task2Grader()
        if task_id == "task3_radicalization_pipeline":
            return Task3Grader()
        return CascadeGrader()

    def reset(self, task_id: str = "task1_health_misinfo", seed: int = 42, n_accounts: Optional[int] = None) -> ResetResult:
        self.task_id = task_id
        self.grader = self._select_grader(task_id)
        self.max_steps = TASK_MAX_STEPS.get(task_id, 20)
        self.community = self.generator.generate(task_id, seed=seed, n_accounts=n_accounts)
        self.step_idx = 0
        self.initialized = True
        self.cumulative_reward = 0.0
        self.actions_taken = []
        self.interventions_made = []
        self._grader_scores = {}
        obs = self._build_observation(None, {})
        return ResetResult(observation=obs, info={"task_id": task_id, "seed": seed})

    def step(self, action: CascadeAction) -> StepResult:
        if not self.initialized or self.community is None:
            raise RuntimeError("Environment not initialized; call reset() first")

        self.step_idx += 1
        last: Dict[str, Any] = {}
        inspected_account = None
        inspected_post = None
        network_neighborhood = None
        spread_trace = None
        cluster_analysis = None

        at = action.action_type

        if at == ActionType.INSPECT_ACCOUNT and action.target_account_id:
            inspected_account = self._snapshot_account(action.target_account_id)
            last = {"ok": True, "detail": "inspected_account", "account_id": action.target_account_id}
        elif at == ActionType.INSPECT_POST and action.target_post_id:
            inspected_post = self._snapshot_post(action.target_post_id)
            last = {"ok": True, "detail": "inspected_post", "post_id": action.target_post_id}
        elif at == ActionType.INSPECT_NETWORK and action.target_account_id:
            network_neighborhood = self._neighborhood(action.target_account_id)
            last = {"ok": True, "detail": "inspected_network"}
        elif at == ActionType.TRACE_SPREAD and action.target_post_id:
            spread_trace = SpreadSimulator.trace_from_post(self.community, action.target_post_id)
            last = {"ok": True, "detail": "trace_spread"}
        elif at == ActionType.ANALYZE_CLUSTER and action.target_cluster_id:
            cluster_analysis = self._analyze_cluster(action.target_cluster_id)
            last = {"ok": True, "detail": "analyze_cluster"}
        elif at == ActionType.SUBMIT_REPORT:
            last = self.community.apply_intervention(action)
        else:
            last = self.community.apply_intervention(action)
            if last.get("ok") and at not in (ActionType.OBSERVE,):
                self.interventions_made.append(
                    {"step": self.step_idx, "action": at.value, "targets": action.model_dump()}
                )

        done = False
        info: Dict[str, Any] = {"last_action": last}

        if at != ActionType.SUBMIT_REPORT:
            self.community.tick(self.step_idx)
        reward = self.grader.compute_step_reward(self.community, action, self.step_idx, self.max_steps)
        bonus = self.grader.terminal_task_bonus(self.community, action) if at == ActionType.SUBMIT_REPORT else None
        if bonus:
            reward.total = max(0.0, min(1.0, reward.total + bonus.total))
            reward.explanation += f" | {bonus.explanation}"

        self.cumulative_reward += reward.total
        reward.cumulative_reward = self.cumulative_reward
        self.actions_taken.append(at.value)

        if at == ActionType.SUBMIT_REPORT:
            done = True
            self._grader_scores["terminal_bonus"] = float(bonus.total) if bonus else 0.0
        if self.step_idx >= self.max_steps:
            done = True
        if self.community.cascade_probability >= 0.97 and self.community.tipping_point_reached:
            done = True
            info["terminated_reason"] = "cascade_critical"

        obs = self._build_observation(
            last,
            {
                "inspected_account": inspected_account,
                "inspected_post": inspected_post,
                "network_neighborhood": network_neighborhood,
                "spread_trace": spread_trace,
                "cluster_analysis": cluster_analysis,
            },
        )
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> StateResult:
        if not self.initialized or self.community is None:
            raise RuntimeError("Environment not initialized")
        s = self.community
        graph_summary = GraphEngine.subgraph_summary(s.graph)
        return StateResult(
            community_graph=graph_summary,
            cascade_probability=s.cascade_probability,
            ground_truth_origins=list(s.origin_accounts),
            tipping_point_step=int(s.tipping_point_step or -1),
            current_step=self.step_idx,
            task_id=self.task_id,
            interventions=list(self.interventions_made),
            grader_scores=dict(self._grader_scores),
        )

    def _status_enum(self, status: str) -> AccountStatus:
        return {
            "active": AccountStatus.ACTIVE,
            "suspended": AccountStatus.SUSPENDED,
            "shadow_banned": AccountStatus.SHADOW_BANNED,
            "removed": AccountStatus.REMOVED,
        }.get(status, AccountStatus.ACTIVE)

    def _risk_from_temp(self, temp: float) -> RiskLevel:
        if temp < 0.35:
            return RiskLevel.LOW
        if temp < 0.55:
            return RiskLevel.MEDIUM
        if temp < 0.75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def _analyze_cluster(self, cluster_id: str) -> Dict[str, Any]:
        st = self.community
        assert st is not None
        member_ids = [n for n in st.graph.nodes if st.graph.nodes[n].get("cluster_id") == cluster_id]
        bridge_ids = GraphEngine.bridge_accounts(st.graph)
        bridges_here = [b for b in bridge_ids if b in member_ids][:8]
        temp = st.cluster_temperatures.get(cluster_id, 0.3)
        return {
            "cluster_id": cluster_id,
            "size": len(member_ids),
            "emotional_temperature": temp,
            "dominant_topics": st.trending_topics[:3],
            "bridge_accounts_here": bridges_here,
            "risk_level": self._risk_from_temp(temp).value,
        }

    def _neighborhood(self, account_id: str) -> List[NetworkEdge]:
        st = self.community
        assert st is not None
        raw = GraphEngine.neighborhood_edges(st.graph, account_id)
        edges = []
        for u, v, data in raw:
            edges.append(
                NetworkEdge(
                    source=u,
                    target=v,
                    relationship=str(data.get("relationship", "interacted")),
                    strength=float(data.get("strength", 0.5)),
                    created_at=str(data.get("created_at", "")),
                )
            )
        return edges

    def _snapshot_account(self, aid: str) -> Optional[AccountSnapshot]:
        st = self.community
        assert st is not None
        a = st.accounts.get(aid)
        if not a:
            return None
        status = self._status_enum(a.status)
        return AccountSnapshot(
            account_id=a.account_id,
            join_date=a.join_date,
            post_count=a.post_count,
            follower_count=a.follower_count,
            following_count=a.following_count,
            engagement_rate=a.engagement_rate,
            recent_topics=list(a.recent_topics),
            posting_frequency=a.posting_frequency,
            network_cluster=a.cluster_id,
            status=status,
            flags=list(a.flags),
        )

    def _snapshot_post(self, pid: str) -> Optional[PostSnapshot]:
        st = self.community
        assert st is not None
        p = st.posts.get(pid)
        if not p:
            return None
        dist = {k: float(v) / max(p.reach, 1) for k, v in p.cluster_reach.items()} if p.cluster_reach else {}
        now = datetime.utcnow()
        velocity = float(p.spread_count) / max(1.0, 1.0)
        return PostSnapshot(
            post_id=p.post_id,
            author_id=p.author_id,
            content_summary=p.content_summary or "summary_unavailable",
            timestamp=p.timestamp or now.isoformat() + "Z",
            engagement=dict(p.engagement) if p.engagement else {"likes": 0, "shares": 0, "comments": 0, "reports": 0},
            spread_velocity=velocity,
            reach=p.reach,
            cluster_distribution=dist,
            flags=list(p.flags),
        )

    def _time_to_cascade_signal(self) -> Optional[str]:
        st = self.community
        assert st is not None
        if st.tipping_point_step is None:
            return None
        remain = st.tipping_point_step - self.step_idx
        if remain <= 2:
            return "imminent"
        if remain <= max(4, st.tipping_point_step // 4):
            return "near"
        return "distant"

    def _cluster_snapshots(self) -> List[CommunityCluster]:
        st = self.community
        assert st is not None
        clusters: Dict[str, List[str]] = {}
        for n in st.graph.nodes:
            cid = str(st.graph.nodes[n].get("cluster_id", "default"))
            clusters.setdefault(cid, []).append(n)
        out = []
        bridge = GraphEngine.bridge_accounts(st.graph)
        for cid, members in clusters.items():
            temp = float(st.cluster_temperatures.get(cid, 0.35))
            out.append(
                CommunityCluster(
                    cluster_id=cid,
                    size=len(members),
                    dominant_topics=st.trending_topics[:2],
                    emotional_temperature=temp,
                    internal_trust=max(0.2, 1.0 - temp),
                    bridge_accounts=[b for b in bridge if b in members][:6],
                    risk_level=self._risk_from_temp(temp),
                )
            )
        return sorted(out, key=lambda c: c.cluster_id)

    def _build_observation(
        self,
        last_action_result: Optional[Dict[str, Any]],
        extra: Dict[str, Any],
    ) -> CommunityObservation:
        st = self.community
        assert st is not None
        return CommunityObservation(
            task_id=self.task_id,
            step=self.step_idx,
            max_steps=self.max_steps,
            time_to_cascade=self._time_to_cascade_signal(),
            total_accounts=st.graph.number_of_nodes(),
            total_posts_today=st.total_posts_today,
            active_clusters=self._cluster_snapshots(),
            overall_health_score=st.overall_health,
            trending_topics=list(st.trending_topics),
            recent_flags=list(st.recent_flags),
            last_action_result=last_action_result,
            inspected_account=extra.get("inspected_account"),
            inspected_post=extra.get("inspected_post"),
            network_neighborhood=extra.get("network_neighborhood"),
            spread_trace=extra.get("spread_trace"),
            cluster_analysis=extra.get("cluster_analysis"),
            actions_taken=list(self.actions_taken),
            interventions_made=list(self.interventions_made),
        )
