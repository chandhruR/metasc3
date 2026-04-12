from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    INSPECT_ACCOUNT = "inspect_account"
    INSPECT_POST = "inspect_post"
    INSPECT_NETWORK = "inspect_network"
    TRACE_SPREAD = "trace_spread"
    ANALYZE_CLUSTER = "analyze_cluster"

    REMOVE_POST = "remove_post"
    DOWNRANK_POST = "downrank_post"
    ADD_FRICTION = "add_friction"
    SUSPEND_ACCOUNT = "suspend_account"
    SHADOW_BAN_ACCOUNT = "shadow_ban_account"
    REMOVE_ACCOUNT = "remove_account"
    DISCONNECT_ACCOUNTS = "disconnect_accounts"

    OBSERVE = "observe"
    SUBMIT_REPORT = "submit_report"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SHADOW_BANNED = "shadow_banned"
    REMOVED = "removed"


class AccountSnapshot(BaseModel):
    account_id: str
    join_date: str
    post_count: int
    follower_count: int
    following_count: int
    engagement_rate: float
    recent_topics: List[str]
    posting_frequency: float
    network_cluster: str
    status: AccountStatus
    flags: List[str]


class PostSnapshot(BaseModel):
    post_id: str
    author_id: str
    content_summary: str
    timestamp: str
    engagement: Dict[str, int]
    spread_velocity: float
    reach: int
    cluster_distribution: Dict[str, float]
    flags: List[str]


class NetworkEdge(BaseModel):
    source: str
    target: str
    relationship: str
    strength: float
    created_at: str


class CommunityCluster(BaseModel):
    cluster_id: str
    size: int
    dominant_topics: List[str]
    emotional_temperature: float
    internal_trust: float
    bridge_accounts: List[str]
    risk_level: RiskLevel


class CommunityObservation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    time_to_cascade: Optional[str]

    total_accounts: int
    total_posts_today: int
    active_clusters: List[CommunityCluster]
    overall_health_score: float
    trending_topics: List[str]
    recent_flags: List[str]

    last_action_result: Optional[Dict[str, Any]]

    inspected_account: Optional[AccountSnapshot]
    inspected_post: Optional[PostSnapshot]
    network_neighborhood: Optional[List[NetworkEdge]]
    spread_trace: Optional[Dict[str, Any]]
    cluster_analysis: Optional[Dict[str, Any]]

    actions_taken: List[str]
    interventions_made: List[Dict[str, Any]]


class CascadeAction(BaseModel):
    action_type: str
    target_account_id: Optional[str] = None
    target_post_id: Optional[str] = None
    target_cluster_id: Optional[str] = None
    target_edge: Optional[Any] = None
    report: Optional[Any] = None
    reasoning: str = ""


class CascadeReward(BaseModel):
    total: float = Field(default=0.5, ge=0.0, le=1.0)

    cascade_prevention: float = 0.1
    intervention_precision: float = 0.1
    community_health: float = 0.1
    early_detection: float = 0.1
    causal_explanation: float = 0.1

    false_positive_penalty: float = 0.1
    overcorrection_penalty: float = 0.1
    too_late_penalty: float = 0.1

    step_reward: float = 0.1
    cumulative_reward: float = 0.1
    explanation: str = ""


class StepResult(BaseModel):
    observation: CommunityObservation
    reward: CascadeReward
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: CommunityObservation
    info: Dict[str, Any]


class StateResult(BaseModel):
    community_graph: Dict[str, Any]
    cascade_probability: float
    ground_truth_origins: List[str]
    tipping_point_step: int
    current_step: int
    task_id: str
    interventions: List[Dict[str, Any]]
    grader_scores: Dict[str, float]
