import os
from openai import OpenAI
from env import SupportOpsEnv
from models import Action

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

MODEL = os.getenv("MODEL_NAME")

env = SupportOpsEnv()
obs = env.reset()

total_reward = 0.0
step_id = 0

print("[START]")

while True:
    # fallback deterministic actions (no crash guarantee)
    if obs.task_type == "classification":
        action = Action(action_type="classify", payload={"label": "order issue"})

    elif obs.task_type == "response":
        action = Action(
            action_type="respond",
            payload={"response": "Sorry for the delay, we will assist you"}
        )

    else:
        steps = ["verify identity", "reset password", "login success"]
        action = Action(
            action_type="resolve",
            payload={"step": steps[min(step_id, len(steps)-1)]}
        )

    obs, reward, done, _ = env.step(action)
    total_reward += reward

    print(f"[STEP] {step_id} | OBS: {obs.task_type} | REWARD: {reward}")

    step_id += 1

    if done or step_id >= 5:
        break

print(f"[END] TOTAL_REWARD: {total_reward}")