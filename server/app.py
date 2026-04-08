from fastapi import FastAPI
import uvicorn
from env import SupportOpsEnv

app = FastAPI()
env = SupportOpsEnv()

@app.get("/")
def root():
    return {"message": "SupportOpsEnv running"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
