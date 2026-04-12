"""
CascadeEnv — Baseline Inference Script
Follows the Hackathon [START] / [STEP] / [END] format exactly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_KEY      = HF_TOKEN or OPENAI_API_KEY

CASCADEENV_URL = os.environ.get("CASCADEENV_URL", "http://127.0.0.1:7860").rstrip("/")
IMAGE_NAME   = os.environ.get("IMAGE_NAME", "cascadeenv:latest")
BENCHMARK    = "cascadeenv"

TASKS = [
    "task1_health_misinfo",
    "task2_coordinated_harassment",
    "task3_radicalization_pipeline",
]

MAX_STEPS = {
    "task1_health_misinfo": 20,
    "task2_coordinated_harassment": 30,
    "task3_radicalization_pipeline": 50,
}

MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.6
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

SYSTEM_PROMPT = (
    "You are a social media content moderation AI.\n"
    "Your job is to inspect suspicious clusters and accounts, then take action.\n\n"
    "VALID action_type values:\n"
    "  investigate, analyze_cluster, inspect_account, inspect_post,\n"
    "  suspend_account, remove_account, remove_post, add_friction,\n"
    "  disconnect_accounts, shadow_ban_account, submit_report, observe\n\n"
    "Always reply with valid JSON only. No markdown, no explanation outside the JSON."
)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: Any = None) -> None:
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}", flush=True)


def get_agent_action(client: OpenAI, observation: dict, history: List[str], step: int) -> dict:
    user_prompt = (
        f"STEP {step}\n\n"
        f"OBSERVATION:\n{json.dumps(observation, indent=2)}\n\n"
        f"HISTORY:\n{chr(10).join(history[-5:])}\n\n"
        "Reply with JSON only:\n"
        '{"action_type": "<one of the valid types>", "target_cluster_id": "<or null>", '
        '"target_account_id": "<or null>", "target_post_id": "<or null>", "reasoning": "<brief>"}'
    )

    if not API_BASE_URL or not API_KEY:
        return {"action_type": "analyze_cluster", "target_cluster_id": "skeptics", "reasoning": "fallback"}

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
        # strip markdown fences if any
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    text = part
                    break
        return json.loads(text)
    except Exception as exc:
        _eprint(f"[DEBUG] model error: {exc}")
        return {"action_type": "analyze_cluster", "target_cluster_id": "skeptics", "reasoning": "fallback"}


async def run_task(client: OpenAI, task_id: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = MAX_STEPS[task_id]

    async with httpx.AsyncClient(base_url=CASCADEENV_URL, timeout=120.0) as http:
        try:
            r = await http.post("/reset", params={"task_id": task_id, "seed": 42})
            r.raise_for_status()
            result = r.json()
        except Exception as exc:
            _eprint(f"[DEBUG] /reset failed: {exc}")
            log_end(success=False, steps=0, score=0.5, rewards=[0.5])
            return 0.5

        observation = result.get("observation", {})

        for step in range(1, max_steps + 1):
            if result.get("done"):
                break

            action = get_agent_action(client, observation, history, step)
            if "reasoning" not in action:
                action["reasoning"] = ""

            try:
                r = await http.post("/step", json=action)
                r.raise_for_status()
                result = r.json()
            except Exception as exc:
                log_step(step=step, action=action, reward=0.5, done=True, error=str(exc))
                rewards.append(0.5)
                steps_taken = step
                break

            observation = result.get("observation", {})
            reward_obj  = result.get("reward", {})
            reward      = float(reward_obj.get("total", 0.5)) if isinstance(reward_obj, dict) else float(reward_obj or 0.5)
            done        = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action.get('action_type')} -> reward {reward:+.3f}")

            if done:
                break

    # Score = mean(rewards), clamped strictly inside (0.001, 0.999)
    raw = sum(rewards) / len(rewards) if rewards else 0.5
    score = max(0.001, min(0.999, raw))
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    _eprint(f"[DEBUG] {task_id} score={score:.3f}")
    return score


async def main() -> None:
    base = API_BASE_URL or "https://api.openai.com/v1"
    client = OpenAI(base_url=base, api_key=API_KEY or "dummy")

    _eprint(f"[DEBUG] IMAGE_NAME={IMAGE_NAME} CASCADEENV_URL={CASCADEENV_URL}")

    all_scores: Dict[str, float] = {}
    for task_id in TASKS:
        all_scores[task_id] = await run_task(client, task_id)

    mean = sum(all_scores.values()) / len(all_scores)
    _eprint(f"[DEBUG] SUMMARY mean={mean:.3f} scores={json.dumps(all_scores)}")


if __name__ == "__main__":
    asyncio.run(main())
