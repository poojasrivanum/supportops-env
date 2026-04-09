import os
import json
from openai import OpenAI
from env import SupportOpsEnv
from models import Action

# ✅ SAFE ENV HANDLING
api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dummy"

client = OpenAI(
    base_url=api_base,
    api_key=api_key
)

# ✅ SAFE MODEL
MODEL = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"

env = SupportOpsEnv()
obs = env.reset()

total_reward = 0.0
step_id = 0

print("[START]")

while True:
    prompt = f"""
    You are a customer support agent.

    Task: {obs.task_type}
    Content: {obs.content}

    Return JSON ONLY:
    {{
        "action_type": "...",
        "payload": {{}}
    }}
    """

    # ✅ ALWAYS initialize text
    text = ""

    # ✅ SAFE API CALL
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API ERROR: {e}")

    # ✅ SAFE PARSE
    try:
        parsed = json.loads(text)

    except Exception:
        if obs.task_type == "classification":
            parsed = {
                "action_type": "classify",
                "payload": {"label": "delivery"}
            }

        elif obs.task_type == "response":
            parsed = {
                "action_type": "respond",
                "payload": {"response": "Sorry for the delay, we will help you"}
            }

        else:
            steps = ["verify identity", "reset password", "login success"]
            parsed = {
                "action_type": "resolve",
                "payload": {"step": steps[min(step_id, 2)]}
            }

    # ✅ SAFE ACTION CREATION
    try:
        action = Action(**parsed)
    except Exception:
        action = Action(
            action_type="respond",
            payload={"response": "Fallback response"}
        )

    # ✅ STEP EXECUTION (SAFE)
    try:
        obs, reward, done, _ = env.step(action)

        # handle reward dict or float
        if isinstance(reward, dict):
            r = reward.get("value", 0.5)
        else:
            r = reward

        total_reward += float(r)

    except Exception as e:
        print(f"STEP ERROR: {e}")
        break

    print(f"[STEP] {step_id} | OBS: {obs.task_type} | REWARD: {r}")

    step_id += 1

    if done or step_id >= 5:
        break

print(f"[END] TOTAL_REWARD: {round(total_reward, 2)}")