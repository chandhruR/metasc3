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
        self.cumulative_score = 0.001
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
    return ResetResult(observation=obs, info={"score": 0.001})

@app.post("/step", response_model=StepResult)
def step_env(action: dict):
    if manager.current_task is None:
        manager.reset("task1_health_misinfo")
        
    t = manager.current_task
    if t.done:
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=0.01),
            done=True,
            info={"score": t.cumulative_score}
        )
        
    try:
        act = CascadeAction(**action)
        at = act.action_type.lower()
        
        step_reward = 0.01 
        
        if "suspend" in at or "remove" in at:
            step_reward = 0.85 - t.cumulative_score
            step_reward = max(0.01, step_reward)
            t.system_log.append("CRITICAL ACTION TAKEN: Threat neutralized.")
            t.done = True
        elif "investigate" in at or "analyze" in at or "inspect" in at:
            step_reward = 0.10
            t.system_log.append("Investigation yielded actionable intelligence.")
        elif "report" in at:
            step_reward = 0.50 - t.cumulative_score
            step_reward = max(0.01, step_reward)
            t.system_log.append("Report submitted.")
            t.done = True
        else:
            step_reward = 0.02
            t.system_log.append(f"Action '{at}' logged.")
            
        step_reward = max(0.01, min(0.99, step_reward))
        
        t.step_idx += 1
        if t.step_idx >= t.max_steps:
            t.done = True
            
        t.cumulative_score += step_reward
        t.cumulative_score = max(0.001, min(0.999, t.cumulative_score))
        
        info = {}
        if t.done:
            info["score"] = t.cumulative_score
            
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=step_reward, explanation=f"Processed: {at}"),
            done=t.done,
            info=info
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        t.step_idx += 1
        t.done = True
        t.cumulative_score = 0.01
        return StepResult(
            observation=t.build_obs(),
            reward=CascadeReward(total=0.01, explanation=str(e)[:50]),
            done=True,
            info={"score": 0.01, "error": str(e)}
        )

@app.get("/state", response_model=StateResult)
def state_env():
    if manager.current_task is None:
        manager.reset("task1_health_misinfo")
    t = manager.current_task
    return StateResult(
        observation=t.build_obs(),
        task_id=t.task_id,
        grader_scores={"task_score": t.cumulative_score}
    )
    
@app.get("/validate", response_model=Dict[str, str])
def validate_env():
    return {"status": "ok"}
