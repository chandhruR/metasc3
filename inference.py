"""
CascadeEnv — Baseline Inference Script
Strictly following the space-separated key=value logging format.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# Required Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
API_KEY      = HF_TOKEN or os.environ.get("API_KEY", "dummy")

CASCADEENV_URL = os.environ.get("CASCADEENV_URL", "http://127.0.0.1:7860").rstrip("/")
IMAGE_NAME   = os.environ.get("IMAGE_NAME", "cascadeenv:latest")

TASK_NAME    = os.environ.get("CASCADEENV_TASK", "task1_health_misinfo")
BENCHMARK    = "cascadeenv"

TASKS = [
    "task1_health_misinfo",
    "task2_coordinated_harassment",
    "task3_radicalization_pipeline",
]

MAX_STEPS_MAP = {
    "task1_health_misinfo": 20,
    "task2_coordinated_harassment": 30,
    "task3_radicalization_pipeline": 50,
}

# Scoring Constants
MAX_TOTAL_REWARD = 1.0 # We ensure sum(rewards) <= 1.0 in env.py
SUCCESS_SCORE_THRESHOLD = 0.4
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

SYSTEM_PROMPT = (
    "You are a content moderation AI. Inspect clusters/accounts and take action.\n"
    "Valid actions: investigate, analyze_cluster, inspect_account, inspect_post,\n"
    "suspend_account, remove_account, remove_post, add_friction,\n"
    "disconnect_accounts, shadow_ban_account, submit_report, observe.\n"
    "Reply with JSON only: {\"action_type\": \"...\", \"reasoning\": \"...\"}"
)


def log_start(task: str, env: str, model: str) -> None:
    # Format: [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Clean action string to avoid breaking space-separated format
    action_clean = action.replace(" ", "_").replace("=", ":")[:50]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.10"
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_agent_action(client: OpenAI, observation: dict, step: int) -> dict:
    user_prompt = f"Step {step}: Observation: {json.dumps(observation)}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception:
        return {"action_type": "analyze_cluster", "reasoning": "fallback"}


async def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = MAX_STEPS_MAP.get(task_id, 20)

    async with httpx.AsyncClient(base_url=CASCADEENV_URL, timeout=120.0) as http:
        try:
            r = await http.post("/reset", params={"task_id": task_id, "seed": 42})
            r.raise_for_status()
            result = r.json()
        except Exception as exc:
            log_end(success=False, steps=0, score=0.1, rewards=[0.10])
            return 0.1

        for step in range(1, max_steps + 1):
            if result.get("done"):
                break

            obs = result.get("observation", {})
            action_dict = get_agent_action(client, obs, step)
            action_type = action_dict.get("action_type", "observe")

            try:
                r = await http.post("/step", json=action_dict)
                r.raise_for_status()
                result = r.json()
            except Exception as exc:
                log_step(step=step, action=action_type, reward=0.01, done=True, error=str(exc))
                rewards.append(0.01)
                steps_taken = step
                break

            reward_obj  = result.get("reward", {})
            reward      = float(reward_obj.get("total", 0.01)) if isinstance(reward_obj, dict) else float(reward_obj or 0.01)
            done        = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_type, reward=reward, done=done, error=None)

            if done:
                break

    # Final score is sum of rewards
    score = sum(rewards)
    score = max(0.001, min(0.999, score))
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        await run_task(client, task_id)


if __name__ == "__main__":
    asyncio.run(main())
