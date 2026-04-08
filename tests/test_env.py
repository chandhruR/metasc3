import pytest

from server.env import CascadeEnvironment
from server.models import ActionType, CascadeAction


@pytest.fixture
def env() -> CascadeEnvironment:
    return CascadeEnvironment()


def test_reset_and_observe(env: CascadeEnvironment) -> None:
    r = env.reset("task1_health_misinfo", seed=1, n_accounts=80)
    assert r.observation.task_id == "task1_health_misinfo"
    assert r.observation.step == 0
    a = CascadeAction(action_type=ActionType.OBSERVE, reasoning="noop")
    s = env.step(a)
    assert s.observation.step == 1
    assert 0.0 <= s.reward.total <= 1.0


def test_submit_ends_episode(env: CascadeEnvironment) -> None:
    env.reset("task1_health_misinfo", seed=2, n_accounts=60)
    action = CascadeAction(
        action_type=ActionType.SUBMIT_REPORT,
        reasoning="closing",
        report={
            "cascade_type": "health misinformation",
            "origin_accounts": env.community.origin_accounts if env.community else [],
            "trigger_post": env.community.trigger_post if env.community else "",
        },
    )
    out = env.step(action)
    assert out.done is True


def test_state_endpoint_hidden_fields(env: CascadeEnvironment) -> None:
    env.reset("task2_coordinated_harassment", seed=3, n_accounts=120)
    st = env.state()
    assert st.task_id == "task2_coordinated_harassment"
    assert isinstance(st.ground_truth_origins, list)
