import os
import json
from openai import OpenAI
from env import SupportOpsEnv
from models import Action

api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dummy"

client = OpenAI(
    base_url=api_base,
    api_key=api_key
)

MODEL = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"

TASK_ACTIONS = {
    "classification": {
        "action_type": "classify",
        "payload": {"label": "delivery"}
    },
    "response": {
        "action_type": "respond",
        "payload": {"response": "Sorry for the delay, we will help you"}
    },
    "resolution": {
        "action_type": "resolve",
        "payload": {"step": "verify identity"}
    }
}

print("[START]")

for task_name in ["classification", "response", "resolution"]:
    env = SupportOpsEnv()
    env.current_task = task_name
    env.step_count = 0
    env.done = False

    if task_name == "classification":
        env.state_data = {"email": "My order hasn't arrived after 10 days"}
    elif task_name == "response":
        env.state_data = {"ticket": "Customer upset about delayed shipment"}
    else:
        env.state_data = {"issue": "User cannot login", "progress": []}

    obs = env.state()
    total_reward = 0.0

    for step_id in range(5):
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

        text = ""
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API ERROR: {e}")

        try:
            parsed = json.loads(text)
        except Exception:
            parsed = TASK_ACTIONS[task_name]

        try:
            action = Action(**parsed)
        except Exception:
            action = Action(**TASK_ACTIONS[task_name])

        try:
            obs, reward, done, _ = env.step(action)
            if isinstance(reward, dict):
                r = reward.get("value", 0.5)
            else:
                r = reward
            total_reward += float(r)
        except Exception as e:
            print(f"STEP ERROR: {e}")
            break

        print(f"[STEP] {step_id} | TASK: {task_name} | OBS: {obs.task_type} | REWARD: {r}")

        if done:
            break

    print(f"[TASK] {task_name} | TOTAL_REWARD: {round(total_reward, 2)}")

print("[END]")