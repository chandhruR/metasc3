from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_KEY = HF_TOKEN or OPENAI_API_KEY

IMAGE_NAME = os.environ.get("IMAGE_NAME", "cascadeenv:latest")
BENCHMARK = "cascadeenv"

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

TEMPERATURE = 0.2
MAX_TOKENS = 1000
SUCCESS_THRESHOLD = 0.6

SYSTEM_PROMPT = (
    "You are doing trust-and-safety work on a simulated social feed. "
    "Inspect before you act, don't ban obvious real users by mistake, "
    "prefer small fixes over mass takedowns, and submit_report when you're done. "
    "Every reply must include reasoning in the JSON."
)


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def log_step(step: int, action: Any, reward: float, done: bool, error: Any = None) -> None:
    payload = {"step": step, "action": action, "reward": reward, "done": done, "error": error}
    print(f"[STEP] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    payload = {"success": success, "steps": steps, "score": score, "rewards": rewards}
    print(f"[END] {json.dumps(payload, ensure_ascii=False)}", flush=True)


def get_agent_action(
    client: OpenAI,
    observation: dict,
    history: List[str],
    step: int,
) -> dict:
    user_prompt = f"""
STEP {step}

CURRENT OBSERVATION:
{json.dumps(observation, indent=2)}

YOUR HISTORY SO FAR:
{chr(10).join(history[-5:])}

Reply with JSON only:
{{
  "action_type": "<action type>",
  "target_account_id": "<optional>",
  "target_post_id": "<optional>",
  "target_cluster_id": "<optional>",
  "target_edge": ["source_id", "target_id"],
  "report": {{}},
  "reasoning": "<short>"
}}

Actions: inspect_account, inspect_post, inspect_network, trace_spread, analyze_cluster,
remove_post, downrank_post, add_friction, suspend_account, shadow_ban_account,
remove_account, disconnect_accounts, observe, submit_report
"""

    if not API_BASE_URL or not API_KEY:
        _eprint("[DEBUG] missing API_BASE_URL or HF_TOKEN/OPENAI_API_KEY")
        return {"action_type": "observe", "reasoning": "missing credentials"}

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{"):
                    text = p
                    break
        return json.loads(text)
    except Exception as exc:
        _eprint(f"[DEBUG] model error: {exc}")
        return {"action_type": "observe", "reasoning": "fallback"}


def _normalize_action_payload(action: dict) -> dict:
    out = dict(action)
    te = out.get("target_edge")
    if isinstance(te, list) and len(te) == 2:
        out["target_edge"] = [str(te[0]), str(te[1])]
    return out


async def run_task(client: OpenAI, task_id: str, env_base_url: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    max_steps = MAX_STEPS[task_id]

    async with httpx.AsyncClient(base_url=env_base_url, timeout=120.0) as http:
        resp = await http.post("/reset", params={"task_id": task_id, "seed": 42})
        resp.raise_for_status()
        result = resp.json()
        observation = result["observation"]

        if result.get("done"):
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        for step in range(1, max_steps + 1):
            action = get_agent_action(client, observation, history, step)
            if "reasoning" not in action:
                action["reasoning"] = ""
            body = _normalize_action_payload(action)

            err: Any = None
            try:
                resp = await http.post("/step", json=body)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                err = str(exc)
                log_step(step=step, action=body, reward=0.0, done=True, error=err)
                break

            observation = result["observation"]
            reward = float(result["reward"]["total"])
            done = bool(result["done"])

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=body, reward=reward, done=done, error=None)

            history.append(f"Step {step}: {action.get('action_type')} -> reward {reward:+.2f}")

            if done:
                break

    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    env_base = os.environ.get("CASCADEENV_URL", "http://127.0.0.1:7860").rstrip("/")
    base = API_BASE_URL or "https://api.openai.com/v1"
    client = OpenAI(base_url=base, api_key=API_KEY or "dummy")

    _eprint(f"[DEBUG] IMAGE_NAME={IMAGE_NAME} CASCADEENV_URL={env_base}")

    all_scores: Dict[str, float] = {}
    for task_id in TASKS:
        score = await run_task(client, task_id, env_base_url=env_base)
        all_scores[task_id] = score
        _eprint(f"[DEBUG] {task_id} score={score:.3f}")

    mean = sum(all_scores.values()) / len(all_scores)
    _eprint(f"[DEBUG] SUMMARY mean={mean:.3f} scores={json.dumps(all_scores)}")


if __name__ == "__main__":
    asyncio.run(main())
