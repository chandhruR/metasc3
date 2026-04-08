from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from server.graders.base import CascadeGrader
from server.models import ActionType, CascadeAction, CascadeReward

if TYPE_CHECKING:
    from server.state import CommunityState


class Task2Grader(CascadeGrader):
    def terminal_task_bonus(
        self, state: "CommunityState", action: CascadeAction
    ) -> Optional[CascadeReward]:
        if action.action_type != ActionType.SUBMIT_REPORT:
            return None

        report = action.report or {}
        bonus = CascadeReward(total=0.0)

        found: Set[str] = set(report.get("origin_accounts") or report.get("coordinated_accounts") or [])
        true_pos = found & state.coordinated_accounts
        false_pos = found & state.organic_harasser_accounts
        if not found and report.get("recommended_interventions"):
            false_pos = set()

        prec = len(true_pos) / max(len(found), 1)
        recall = len(true_pos) / max(len(state.coordinated_accounts), 1)
        f1 = 0.0 if (prec + recall) == 0 else 2 * prec * recall / (prec + recall)
        bonus.intervention_precision += 0.35 * f1
        bonus.cascade_prevention += 0.25 * recall

        if false_pos:
            bonus.false_positive_penalty += min(0.25, len(false_pos) * 0.06)

        tgt = report.get("trigger_post") or report.get("target_account")
        if state.harassment_target_id and str(tgt) == state.harassment_target_id:
            bonus.early_detection += 0.15

        if state.tipping_point_step and not state.tipping_point_reached:
            bonus.cascade_prevention += 0.10
        else:
            bonus.too_late_penalty += 0.08

        pos = (
            bonus.cascade_prevention
            + bonus.intervention_precision
            + bonus.community_health
            + bonus.early_detection
            + bonus.causal_explanation
        )
        neg = bonus.false_positive_penalty + bonus.too_late_penalty
        bonus.total = max(0.0, min(0.5, pos - neg))
        bonus.explanation = f"task2_terminal_f1={f1:.2f}"
        return bonus
