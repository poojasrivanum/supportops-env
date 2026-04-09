import random
from typing import Tuple, Dict, Any
from models import Observation, Action


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
            self.state_data = {
                "email": "My order hasn't arrived after 10 days"
            }

        elif self.current_task == "response":
            self.state_data = {
                "ticket": "Customer upset about delayed shipment"
            }

        else:
            self.state_data = {
                "issue": "User cannot login",
                "progress": []
            }

        return self.state()

    def state(self) -> Observation:
        return Observation(
            task_type=self.current_task,
            content=self.state_data,
            step_count=self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            return self.state(), 0.0, True, {}

        self.step_count += 1
        reward = 0.0

        # EASY
        if self.current_task == "classification":
            label = action.payload.get("label", "").lower()
            if "order" in label or "shipping" in label:
                reward = 1.0
                self.done = True
            else:
                reward = 0.3

        # MEDIUM
        elif self.current_task == "response":
            text = action.payload.get("response", "").lower()
            if "sorry" in text:
                reward += 0.4
            if "delay" in text:
                reward += 0.3
            if "help" in text or "assist" in text:
                reward += 0.3
            self.done = True

        # HARD
        else:
            step = action.payload.get("step", "").lower()
            self.state_data["progress"].append(step)

            if "verify" in step:
                reward += 0.3
            if "reset password" in step:
                reward += 0.4
            if "login success" in step:
                reward += 0.3
                self.done = True

        # penalty
        reward -= 0.03 * self.step_count
        reward = max(0.05, min(0.95, reward))

        return self.state(), reward, self.done, {}