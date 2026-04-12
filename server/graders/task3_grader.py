from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from server.graders.base import CascadeGrader
from server.models import ActionType, CascadeAction, CascadeReward

if TYPE_CHECKING:
    from server.state import CommunityState


class Task3Grader(CascadeGrader):
    def terminal_task_bonus(
        self, state: "CommunityState", action: CascadeAction
    ) -> Optional[CascadeReward]:
        if action.action_type != ActionType.SUBMIT_REPORT:
            return None

        report = action.report or {}
        bonus = CascadeReward(total=0.0)

        bridges_reported: Set[str] = set(report.get("bridge_accounts") or report.get("origin_accounts") or [])
        gateway_hit = len(bridges_reported & state.gateway_accounts) / max(len(state.gateway_accounts), 1)
        bonus.intervention_precision += 0.30 * gateway_hit

        if report.get("cascade_type") and "radical" in str(report.get("cascade_type")).lower():
            bonus.causal_explanation += 0.12

        structural = str(report.get("recommended_interventions") or "").lower()
        if any(k in structural for k in ("disconnect", "bridge", "friction", "gateway")):
            bonus.cascade_prevention += 0.18

        if state.radicalization_accelerated:
            bonus.false_positive_penalty += 0.12

        if state.overall_health >= 0.55:
            bonus.community_health += 0.12

        if state.tipping_point_step and not state.tipping_point_reached:
            bonus.early_detection += 0.14
        else:
            bonus.too_late_penalty += 0.10

        pos = (
            bonus.cascade_prevention
            + bonus.intervention_precision
            + max(0.0, bonus.community_health)
            + bonus.early_detection
            + bonus.causal_explanation
        )
        neg = bonus.false_positive_penalty + bonus.too_late_penalty
        bonus.explanation = f"task3_terminal_bridges={gateway_hit:.2f}"
        return bonus
