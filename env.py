import random
from typing import Tuple, Dict, Any
from models import Observation, Action


def grader_classification(obs, action) -> float:
    label = action.payload.get("label", "").lower()
    if "order" in label or "shipping" in label or "delivery" in label:
        reward = 0.85
    else:
        reward = 0.25
    return reward


def grader_response(obs, action) -> float:
    text = action.payload.get("response", "").lower()
    reward = 0.1
    if "sorry" in text:
        reward += 0.3
    if "delay" in text:
        reward += 0.25
    if "help" in text or "assist" in text:
        reward += 0.25
    return min(0.95, max(0.05, reward))


def grader_resolution(obs, action) -> float:
    step = action.payload.get("step", "").lower()
    reward = 0.1
    if "verify" in step:
        reward += 0.25
    if "reset password" in step:
        reward += 0.35
    if "login success" in step:
        reward += 0.25
    return min(0.95, max(0.05, reward))


GRADERS = {
    "classification": grader_classification,
    "response": grader_response,
    "resolution": grader_resolution,
}


class SupportOpsEnv:
    def __init__(self):
        self.tasks = ["classification", "response", "resolution"]
        self.current_task = None
        self.step_count = 0
        self.max_steps = 5
        self.done = False
        self.state_data = {}

    def reset(self) -> Observation:
        self.current_task = random.choice(self.tasks)
        self.step_count = 0
        self.done = False

        if self.current_task == "classification":
            self.state_data = {"email": "My order hasn't arrived after 10 days"}
        elif self.current_task == "response":
            self.state_data = {"ticket": "Customer upset about delayed shipment"}
        else:
            self.state_data = {"issue": "User cannot login", "progress": []}

        return self.state()

    def state(self) -> Observation:
        return Observation(
            task_type=self.current_task,
            content=self.state_data,
            step_count=self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return self.state(), 0.05, True, {}

        self.step_count += 1

        grader = GRADERS.get(self.current_task)
        reward = grader(self.state(), action) if grader else 0.5

        reward -= 0.02 * self.step_count
        reward = max(0.05, min(0.95, reward))

        if self.current_task == "classification":
            label = action.payload.get("label", "").lower()
            if "order" in label or "shipping" in label or "delivery" in label:
                self.done = True
        elif self.current_task == "response":
            self.done = True
        else:
            step_val = action.payload.get("step", "").lower()
            self.state_data["progress"].append(step_val)
            if "login success" in step_val:
                self.done = True

        if self.step_count >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done, {}