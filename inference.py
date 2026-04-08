import os
from openai import OpenAI
from env import SupportOpsEnv

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)

MODEL = os.getenv("MODEL_NAME")

env = SupportOpsEnv()

print("[START]")

obs = env.reset()
done = False
total_reward = 0
step_id = 0

while not done:
    print(f"[STEP] {step_id} | OBS: {obs}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": str(obs)}]
    )

    reply = response.choices[0].message.content

    action = {
        "action_type": "respond",
        "content": {"text": reply}
    }

    obs, reward, done, _ = env.step(action)

    print(f"[STEP] {step_id} | REWARD: {reward}")

    total_reward = round(total_reward + reward["value"], 2)
    step_id += 1

print(f"[END] TOTAL_REWARD: {round(total_reward, 2)}")

