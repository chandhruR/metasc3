from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class CascadeAction(BaseModel):
    action_type: str = "observe"
    target_account_id: Optional[str] = None
    target_post_id: Optional[str] = None
    target_cluster_id: Optional[str] = None
    reasoning: str = ""

class CascadeReward(BaseModel):
    total: float = Field(default=0.01, ge=0.0, le=1.0)
    explanation: str = ""

class CommunityObservation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    active_clusters: List[Dict[str, Any]]
    recently_flagged_accounts: List[str]
    system_log: List[str]

class StepResult(BaseModel):
    observation: CommunityObservation
    reward: CascadeReward
    done: bool
    info: Dict[str, Any]

class ResetResult(BaseModel):
    observation: CommunityObservation
    info: Dict[str, Any]

class StateResult(BaseModel):
    observation: CommunityObservation
    task_id: str
    grader_scores: Dict[str, float]
