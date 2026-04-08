from server.graders.task1_grader import Task1Grader
from server.models import ActionType, CascadeAction
from server.scenarios.generator import CommunityGenerator


def test_task1_terminal_bonus() -> None:
    gen = CommunityGenerator()
    state = gen.generate_task1(seed=7, n_accounts=50)
    g = Task1Grader()
    action = CascadeAction(
        action_type=ActionType.SUBMIT_REPORT,
        reasoning="The vaccine misinfo post spreads from the new account.",
        report={
            "cascade_type": "health misinformation",
            "origin_accounts": state.origin_accounts,
            "trigger_post": state.trigger_post,
        },
    )
    bonus = g.terminal_task_bonus(state, action)
    assert bonus is not None
    assert bonus.total >= 0.0
