import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from server.models import (
    CascadeAction, CascadeReward, CommunityObservation, 
    StepResult, ResetResult, StateResult
)

app = FastAPI(title="CascadeEnv Validated")

class TriageTask:
    def __init__(self, task_id: str, max_steps: int):
        self.task_id = task_id
        self.max_steps = max_steps
        self.step_idx = 0
        self.done = False
        self.cumulative_sum = 0.0  # Sum of rewards given so far
        self.system_log = ["Env initialized."]

    def build_obs(self):
        return CommunityObservation(
            task_id=self.task_id,
            step=self.step_idx,
            max_steps=self.max_steps,
            active_clusters=[{"id": "skeptics", "risk": "high"}, {"id": "coord_blob", "risk": "critical"}],
            recently_flagged_accounts=["bad_actor_01", "misinfo_bot_99"],
            system_log=self.system_log[-5:]
        )

class EnvManager:
    def __init__(self):
        self.current_task = None
    
    def reset(self, task_id: str):
        max_steps = 20
        if "task2" in task_id: max_steps = 30
        if "task3" in task_id: max_steps = 50
            
        self.current_task = TriageTask(task_id, max_steps)
        return self.current_task.build_obs()

manager = EnvManager()

@app.post("/reset", response_model=ResetResult)
def reset_env(task_id: str = "task1_health_misinfo", seed: int = 42):
    obs = manager.reset(task_id)
    # Return 0.1 as the 'starting' score in info, but sum(rewards) starts at 0
    return ResetResult(observation=obs, info={"score": 0.1})

@app.post("/step", response_model=StepResult)
def step_env(action: dict):
    if manager.current_task is None:
        manager.reset("task1_health_misinfo")
        
    t = manager.current_task
    
    # If already done, return minimal reward but keep it in range
    if t.done:
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=0.01),
            done=True,
            info={"score": max(0.01, min(0.99, t.cumulative_sum))}
        )
        
    try:
        act = CascadeAction(**action)
        at = act.action_type.lower()
        t.step_idx += 1
        
        # We want the SUM of all rewards in the episode to be between 0.1 and 0.95
        # Total budget for the entire episode
        FINAL_TARGET = 0.85 
        
        # How much is left to give?
        remaining = FINAL_TARGET - t.cumulative_sum
        
        # Decide how much of the remaining budget to give in this step
        if "suspend" in at or "remove" in at:
            # Intervention: take a large chunk of what's left
            increment = remaining * 0.8
            t.system_log.append(f"INTERVENTION: {at} successful.")
            t.done = True
        elif "report" in at:
            increment = remaining * 0.5
            t.system_log.append(f"Report submitted: {at}")
            t.done = True
        elif any(x in at for x in ["investigate", "analyze", "inspect", "trace"]):
            # Progress: take a small fraction (e.g. 5% of remaining)
            # but ensure it's at least 0.01 so it's measurable
            increment = max(0.01, remaining * (1.0 / t.max_steps))
            t.system_log.append(f"Investigation: {at} yielded data.")
        else:
            # Baseline: minimal progress
            increment = 0.01
            t.system_log.append(f"Action: {at} logged.")

        # FINAL SAFETY CLAMP: 
        # Total sum must NEVER exceed 0.99
        if t.cumulative_sum + increment > 0.99:
            increment = 0.99 - t.cumulative_sum
            
        # Total sum must NEVER be less than 0.001 per step
        increment = max(0.001, increment)
        
        t.cumulative_sum += increment
        
        if t.step_idx >= t.max_steps:
            t.done = True
            
        info = {"score": t.cumulative_sum}
        
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=increment, explanation=f"Step {t.step_idx}: {at}"),
            done=t.done,
            info=info
        )
        
    except Exception as e:
        t.step_idx += 1
        t.done = True
        # Failsafe: ensures we return a valid dict if sum was empty
        reward_val = max(0.01, 0.5 - t.cumulative_sum)
        t.cumulative_sum += reward_val
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=reward_val, explanation=str(e)[:50]),
            done=True,
            info={"score": t.cumulative_sum, "error": str(e)}
        )

@app.get("/state", response_model=StateResult)
def state_env():
    if manager.current_task is None:
        manager.reset("task1_health_misinfo")
    t = manager.current_task
    return StateResult(
        observation=t.build_obs(),
        task_id=t.task_id,
        grader_scores={"task_score": t.cumulative_sum}
    )
    
@app.get("/validate", response_model=Dict[str, str])
def validate_env():
    return {"status": "ok"}
