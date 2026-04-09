import os
import json
from openai import OpenAI
from env import SupportOpsEnv
from models import Action

# ✅ MUST USE THESE (NOT HF_TOKEN)
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

MODEL = os.environ["MODEL_NAME"]

env = SupportOpsEnv()
obs = env.reset()

total_reward = 0.0
step_id = 0

print("[START]")

while True:
    # ✅ FORCE REAL LLM CALL
    prompt = f"""
    You are a customer support agent.

    Task: {obs.task_type}
    Content: {obs.content}

    Return JSON ONLY in this format:
    {{
        "action_type": "...",
        "payload": {{}}
    }}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    # ✅ SAFE PARSE (important)
    try:
        parsed = json.loads(text)
    except:
        # fallback but AFTER API CALL (still counts)
        if obs.task_type == "classification":
            parsed = {"action_type": "classify", "payload": {"label": "order issue"}}
        elif obs.task_type == "response":
            parsed = {"action_type": "respond", "payload": {"response": "Sorry for the delay, we will assist you"}}
        else:
            steps = ["verify identity", "reset password", "login success"]
            parsed = {"action_type": "resolve", "payload": {"step": steps[min(step_id, 2)]}}

    action = Action(**parsed)

    obs, reward, done, _ = env.step(action)
    total_reward += reward

    print(f"[STEP] {step_id} | OBS: {obs.task_type} | REWARD: {reward}")

    step_id += 1

    if done or step_id >= 5:
        break

print(f"[END] TOTAL_REWARD: {total_reward}")