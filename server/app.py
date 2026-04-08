import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from client import Client

app = FastAPI()
client = Client()

@app.post("/reset")
def reset():
    return client.reset()

@app.post("/step")
def step(action: dict):
    return client.step(action)

@app.get("/state")
def state():
    return client.state()
@app.get("/")
def root():
    return {
        "status": "SupportOpsEnv running",
        "message": "Use /reset, /step, /state endpoints"
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()