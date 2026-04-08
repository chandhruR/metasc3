from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

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

    if not API_BASE_URL or not HF_TOKEN:
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


def _cluster_pick(observation: dict) -> str | None:
    clusters = observation.get("active_clusters") or []
    if not clusters:
        return None
    def score(c: dict) -> float:
        risk = str(c.get("risk_level", "")).lower()
        risk_w = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.0}.get(risk, 0.5)
        temp = float(c.get("emotional_temperature") or 0.0)
        return risk_w + temp
    best = max(clusters, key=score)
    return best.get("cluster_id")


def _make_report(observation: dict) -> dict:
    task_id = observation.get("task_id", "")
    inspected_post = observation.get("inspected_post") or {}
    inspected_account = observation.get("inspected_account") or {}
    cluster_analysis = observation.get("cluster_analysis") or {}

    cascade_type = {
        "task1_health_misinfo": "health_misinformation",
        "task2_coordinated_harassment": "coordinated_harassment",
        "task3_radicalization_pipeline": "radicalization_pipeline",
    }.get(task_id, "unknown")

    origin_accounts = []
    if inspected_account.get("account_id"):
        origin_accounts.append(inspected_account["account_id"])
    elif (cluster_analysis.get("bridge_accounts_here") or cluster_analysis.get("bridge_accounts_here")):
        origin_accounts = list(cluster_analysis.get("bridge_accounts_here") or [])[:3]

    trigger_post = inspected_post.get("post_id") or ""

    return {
        "cascade_type": cascade_type,
        "origin_accounts": origin_accounts,
        "trigger_post": trigger_post,
        "vulnerability": "insufficient signal to be confident; report based on partial inspections",
        "recommended_interventions": ["add_friction", "downrank_post", "inspect_network"],
        "predicted_outcome_without_intervention": "cascade probability likely increases with time pressure",
    }


def _force_action(observation: dict, step: int, max_steps: int) -> dict | None:
    deadline = max(3, int(max_steps * 0.7))
    if step >= deadline:
        return {
            "action_type": "submit_report",
            "report": _make_report(observation),
            "reasoning": "Submitting report before deadline to avoid stalling.",
        }

    if step <= 3:
        cid = _cluster_pick(observation)
        if cid:
            return {
                "action_type": "analyze_cluster",
                "target_cluster_id": cid,
                "reasoning": "Early investigation: analyze highest-risk cluster.",
            }

    ca = observation.get("cluster_analysis") or {}
    bridges = ca.get("bridge_accounts_here") or ca.get("bridge_accounts_here") or ca.get("bridge_accounts") or []
    if bridges:
        return {
            "action_type": "inspect_account",
            "target_account_id": str(bridges[0]),
            "reasoning": "Inspecting a bridge/gateway account surfaced by cluster analysis.",
        }

    ia = observation.get("inspected_account") or {}
    if ia.get("account_id"):
        return {
            "action_type": "inspect_network",
            "target_account_id": ia["account_id"],
            "reasoning": "Network context often reveals spread/coordination.",
        }

    cid = _cluster_pick(observation)
    if cid:
        return {
            "action_type": "analyze_cluster",
            "target_cluster_id": cid,
            "reasoning": "Need more signal; analyzing cluster for bridge accounts.",
        }

    return None


async def run_task(client: OpenAI, task_id: str, env_base_url: str) -> float:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    max_steps = MAX_STEPS[task_id]
    observe_streak = 0
    analyze_streak = 0
    last_analyzed_cluster: str | None = None

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

            at = str(action.get("action_type", "")).lower()
            if at == "observe":
                observe_streak += 1
            else:
                observe_streak = 0

            if at == "analyze_cluster":
                cid = action.get("target_cluster_id")
                if cid and cid == last_analyzed_cluster:
                    analyze_streak += 1
                else:
                    analyze_streak = 1
                    last_analyzed_cluster = cid
            else:
                analyze_streak = 0
                last_analyzed_cluster = None

            forced = None
            deadline = max(3, int(max_steps * 0.7))
            if observe_streak >= 2 or analyze_streak >= 3 or step <= 3 or step >= deadline:
                forced = _force_action(observation, step, max_steps)

            if forced:
                action = {**action, **forced}
                if forced.get("action_type") != "observe":
                    observe_streak = 0

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
    if not HF_TOKEN:
        print("HF_TOKEN is required", file=sys.stderr, flush=True)
        raise SystemExit(2)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
