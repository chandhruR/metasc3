from fastapi import FastAPI, HTTPException

from server.env import CascadeEnvironment
from server.models import CascadeAction, ResetResult, StateResult, StepResult

app = FastAPI(title="CascadeEnv", version="1.0.0")
env = CascadeEnvironment()


@app.post("/reset", response_model=ResetResult)
async def reset(task_id: str = "task1_health_misinfo", seed: int = 42, n_accounts: int | None = None):
    return env.reset(task_id=task_id, seed=seed, n_accounts=n_accounts)


@app.post("/step", response_model=StepResult)
async def step(action: CascadeAction):
    if not env.initialized:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.step(action)


@app.get("/state", response_model=StateResult)
async def state():
    if not env.initialized:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


@app.get("/validate")
async def validate():
    return {"status": "ok", "tasks": 3, "spec": "openenv-v1"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
