import os
import json
import re
from openai import OpenAI
from env import SupportOpsEnv
from models import Action

# --- Setup ---
api_base = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=api_base,
    api_key=api_key
)

MODEL = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"

env = SupportOpsEnv()
obs = env.reset()

total_reward = 0.0
step_id = 0

print("[START]")


# --- Helper: Extract JSON safely ---
def extract_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No valid JSON found")


while True:
    prompt = f"""
You are a customer support agent.

Task: {obs.task_type}
Content: {obs.content}

Return ONLY valid JSON. No explanation, no markdown.

Format:
{{
    "action_type": "...",
    "payload": {{}}
}}
"""

    text = ""

    # --- Safe API Call ---
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        if not response or not response.choices:
            raise ValueError("Empty response")

        text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[API ERROR] {e}")
        text = ""

    # --- Safe Parsing ---
    try:
        parsed = extract_json(text)

        if "action_type" not in parsed or "payload" not in parsed:
            raise ValueError("Invalid response format")

    except Exception as e:
        print(f"[PARSE ERROR] Raw output: {text}")

        # --- Fallbacks ---
        if obs.task_type == "classification":
            parsed = {
                "action_type": "classify",
                "payload": {"label": "delivery"}
            }

        elif obs.task_type == "response":
            parsed = {
                "action_type": "respond",
                "payload": {"response": "Sorry for the inconvenience."}
            }

        else:
            steps = ["verify identity", "reset password", "login success"]
            parsed = {
                "action_type": "resolve",
                "payload": {"step": steps[min(step_id, 2)]}
            }

    # --- Execute Action ---
    try:
        action = Action(**parsed)
        obs, reward, done, _ = env.step(action)

    except Exception as e:
        print(f"[ENV ERROR] {e}")
        break

    total_reward += reward

    print(f"[STEP] {step_id} | TASK: {obs.task_type} | REWARD: {reward}")

    step_id += 1

    if done or step_id >= 5:
        break


print(f"[END] TOTAL_REWARD: {total_reward}")