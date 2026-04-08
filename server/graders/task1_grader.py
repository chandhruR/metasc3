from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from server.graders.base import CascadeGrader
from server.models import ActionType, CascadeAction, CascadeReward

if TYPE_CHECKING:
    from server.state import CommunityState


class Task1Grader(CascadeGrader):
    def terminal_task_bonus(
        self, state: "CommunityState", action: CascadeAction
    ) -> Optional[CascadeReward]:
        if action.action_type != ActionType.SUBMIT_REPORT:
            return None
        report = action.report or {}
        bonus = CascadeReward(total=0.0)

        origins = set(state.origin_accounts)
        reported_origins = set(report.get("origin_accounts") or [])
        if reported_origins and origins == reported_origins:
            bonus.community_health += 0.25
        elif origins & reported_origins:
            bonus.community_health += 0.12

        if report.get("trigger_post") == state.trigger_post:
            bonus.intervention_precision += 0.20
        elif state.trigger_post in str(report.get("trigger_post", "")):
            bonus.intervention_precision += 0.10

        if state.tipping_point_step and not state.tipping_point_reached:
            bonus.cascade_prevention += 0.20
        elif state.tipping_point_step and state.tipping_point_reached:
            bonus.too_late_penalty += 0.10

        innocent = len(state.actioned_real_users) + len({p for p in state.wrongly_removed_posts})
        if innocent == 0:
            bonus.early_detection += 0.20
        else:
            bonus.false_positive_penalty += min(0.20, innocent * 0.05)

        ctype = report.get("cascade_type", "")
        if ctype and ("misinfo" in ctype.lower() or "health" in ctype.lower()):
            bonus.causal_explanation += 0.15

        pos = (
            bonus.cascade_prevention
            + bonus.intervention_precision
            + bonus.community_health
            + bonus.early_detection
            + bonus.causal_explanation
        )
        neg = bonus.false_positive_penalty + bonus.too_late_penalty
        bonus.total = max(0.0, min(0.45, pos - neg))
        bonus.explanation = f"task1_terminal_bonus={bonus.total:.3f}"
        return bonus
