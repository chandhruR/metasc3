import sys
from server.env import CascadeEnvironment
from server.models import CascadeAction, ActionType

env = CascadeEnvironment()
env.reset("task3_radicalization_pipeline")
for step in range(1, 35):
    # simulate the previous actions
    act = CascadeAction(action_type=ActionType.OBSERVE, reasoning="tick")
    res = env.step(act)

# Now, Step 35: add_friction
act = CascadeAction(action_type=ActionType.ADD_FRICTION, target_cluster_id="high_intensity", reasoning="test")
res = env.step(act)
print(f"Step 35 reward: {res.reward.total}")
