from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

from server.models import ActionType, CascadeAction, CascadeReward

if TYPE_CHECKING:
    from server.state import CommunityState


class CascadeGrader:
    WEIGHTS = {
        "cascade_prevention": 0.25,
        "intervention_precision": 0.20,
        "community_health": 0.20,
        "early_detection": 0.20,
        "causal_explanation": 0.15,
        "false_positive_penalty": 0.30,
        "overcorrection_penalty": 0.20,
        "too_late_penalty": 0.15,
    }

    def compute_step_reward(
        self,
        state: "CommunityState",
        action: CascadeAction,
        step: int,
        max_steps: int,
    ) -> CascadeReward:
        reward = CascadeReward()

        delta_cascade = self._measure_cascade_delta(state, action)
        reward.cascade_prevention = max(0.0, delta_cascade) * self.WEIGHTS["cascade_prevention"]

        precision = self._measure_precision(state, action)
        reward.intervention_precision = precision * self.WEIGHTS["intervention_precision"]

        health_delta = self._measure_health_delta(state, action)
        reward.community_health = max(0.0, health_delta) * self.WEIGHTS["community_health"]

        if action.action_type == ActionType.SUBMIT_REPORT and state.tipping_point_step:
            steps_before = max(0, state.tipping_point_step - step)
            early_bonus = steps_before / max(state.tipping_point_step, 1)
            reward.early_detection = min(1.0, early_bonus) * self.WEIGHTS["early_detection"]

        if action.reasoning:
            explanation_score = self._grade_explanation(action.reasoning, state)
            reward.causal_explanation = explanation_score * self.WEIGHTS["causal_explanation"]

        false_positives = state.check_false_positives(action)
        if false_positives:
            penalty = min(len(false_positives) * 0.05, self.WEIGHTS["false_positive_penalty"])
            reward.false_positive_penalty = penalty

        overcorrection = self._measure_overcorrection(state, action)
        reward.overcorrection_penalty = overcorrection * self.WEIGHTS["overcorrection_penalty"]

        if state.tipping_point_reached and action.action_type != ActionType.SUBMIT_REPORT:
            reward.too_late_penalty = self.WEIGHTS["too_late_penalty"] * min(
                1.0, (state.cascade_probability / max(0.05, 1e-6))
            )

        positive = (
            reward.cascade_prevention
            + reward.intervention_precision
            + reward.community_health
            + reward.early_detection
            + reward.causal_explanation
        )
        negative = (
            reward.false_positive_penalty
            + reward.overcorrection_penalty
            + reward.too_late_penalty
        )
        reward.total = max(0.0, min(1.0, positive - negative))
        reward.step_reward = reward.total
        reward.explanation = self._explain_reward(reward)
        return reward

    def _measure_cascade_delta(self, state: "CommunityState", action: CascadeAction) -> float:
        drop = state.prev_cascade_probability - state.cascade_probability
        if action.action_type in (
            ActionType.REMOVE_POST,
            ActionType.DOWNRANK_POST,
            ActionType.ADD_FRICTION,
            ActionType.SUSPEND_ACCOUNT,
            ActionType.REMOVE_ACCOUNT,
            ActionType.SHADOW_BAN_ACCOUNT,
            ActionType.DISCONNECT_ACCOUNTS,
        ):
            return max(0.0, drop + 0.05)
        if action.action_type == ActionType.SUBMIT_REPORT:
            return max(0.0, drop)
        return max(0.0, drop * 0.5)

    def _measure_precision(self, state: "CommunityState", action: CascadeAction) -> float:
        at = action.action_type
        if at in (ActionType.OBSERVE, ActionType.SUBMIT_REPORT):
            return 0.35
        if at in (ActionType.INSPECT_ACCOUNT, ActionType.INSPECT_POST, ActionType.INSPECT_NETWORK):
            return 0.55
        if at in (
            ActionType.TRACE_SPREAD,
            ActionType.ANALYZE_CLUSTER,
        ):
            return 0.6

        if state.cascade_type == "health_misinformation":
            if at == ActionType.REMOVE_POST and action.target_post_id == state.trigger_post:
                return 1.0
            if at == ActionType.ADD_FRICTION and action.target_post_id == state.trigger_post:
                return 0.85
            if at in (ActionType.SUSPEND_ACCOUNT, ActionType.REMOVE_ACCOUNT) and action.target_account_id:
                if action.target_account_id in state.origin_accounts:
                    return 0.95
                return 0.25

        if state.cascade_type == "coordinated_harassment":
            if at in (ActionType.SUSPEND_ACCOUNT, ActionType.REMOVE_ACCOUNT, ActionType.SHADOW_BAN_ACCOUNT):
                aid = action.target_account_id or ""
                if aid in state.coordinated_accounts:
                    return 1.0
                if aid in state.organic_harasser_accounts or aid == state.harassment_target_id:
                    return 0.15
                return 0.35

        if state.cascade_type == "radicalization_pipeline":
            if at == ActionType.DISCONNECT_ACCOUNTS:
                s = action.target_edge[0] if action.target_edge else ""
                t = action.target_edge[1] if action.target_edge else ""
                if s in state.gateway_accounts or t in state.gateway_accounts:
                    return 0.9
                return 0.35
            if at == ActionType.ADD_FRICTION and action.target_post_id:
                return 0.55

        return 0.45

    def _measure_health_delta(self, state: "CommunityState", action: CascadeAction) -> float:
        d = state.overall_health - state.prev_overall_health
        return max(-1.0, min(1.0, d * 4.0))

    def _measure_overcorrection(self, state: "CommunityState", action: CascadeAction) -> float:
        if state.intervention_count < 6:
            return 0.0
        severe = state.intervention_count / 40.0
        innocent_hits = len(state.actioned_real_users) + len(state.wrongly_removed_posts)
        return min(1.0, severe + innocent_hits * 0.08)

    def _grade_explanation(self, reasoning: str, state: "CommunityState") -> float:
        text = reasoning.lower()
        if len(text.strip()) < 12:
            return 0.15

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                return float(self._llm_grade_explanation(reasoning, state))
            except Exception:
                pass
        return self._heuristic_explanation_score(text, state)

    def _heuristic_explanation_score(self, text: str, state: "CommunityState") -> float:
        score = 0.35
        if state.cascade_type == "health_misinformation":
            if any(k in text for k in ("vaccine", "misinfo", "health", "post", "spread")):
                score += 0.2
            if any(o in text for o in state.origin_accounts):
                score += 0.25
        elif state.cascade_type == "coordinated_harassment":
            if any(k in text for k in ("coordinat", "timing", "pattern", "bot", "pile")):
                score += 0.25
            if state.harassment_target_id and state.harassment_target_id in text:
                score += 0.15
        elif state.cascade_type == "radicalization_pipeline":
            if any(k in text for k in ("bridge", "cluster", "gateway", "gradual", "pathway")):
                score += 0.25
        if len(text) > 120:
            score += 0.1
        return float(min(1.0, score))

    def _llm_grade_explanation(self, reasoning: str, state: "CommunityState") -> float:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            f"Score this moderation reasoning 0.0-1.0. Context: origins={state.origin_accounts}, "
            f"type={state.cascade_type}\n{reasoning}\nReply with one number only."
        )
        resp = client.chat.completions.create(
            model=os.environ.get("OPENENV_GRADER_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        )
        raw = resp.choices[0].message.content.strip()
        return max(0.0, min(1.0, float(raw.split()[0])))

    def _explain_reward(self, reward: CascadeReward) -> str:
        parts = [
            f"+cascade_prev={reward.cascade_prevention:.3f}",
            f"+precision={reward.intervention_precision:.3f}",
            f"+health={reward.community_health:.3f}",
            f"+early={reward.early_detection:.3f}",
            f"+explain={reward.causal_explanation:.3f}",
            f"-fp={reward.false_positive_penalty:.3f}",
            f"-overcorr={reward.overcorrection_penalty:.3f}",
            f"-late={reward.too_late_penalty:.3f}",
        ]
        return "; ".join(parts)

    def terminal_task_bonus(
        self, state: "CommunityState", action: CascadeAction
    ) -> Optional[CascadeReward]:
        return None
