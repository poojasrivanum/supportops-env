import random

class SupportOpsEnv:
    def __init__(self):
        self.tasks = [
            {
                "id": "easy",
                "text": "My order has not arrived",
                "type": "classification",
                "expected": {
                    "category": "delivery",
                    "priority": "high"
                }
            },
            {
                "id": "medium",
                "text": "I want a refund, it's delayed",
                "type": "response",
                "expected_keywords": ["sorry", "refund", "delay"]
            },
            {
                "id": "hard",
                "text": "I was charged twice and I am angry",
                "type": "resolution",
                "expected_keywords": ["refund", "escalate", "double charge"]
            }
        ]

        self.current_task = None
        self.step_count = 0

    def reset(self):
        self.current_task = random.choice(self.tasks)
        self.step_count = 0

        return {
            "task_id": self.current_task["id"],
            "input_text": self.current_task["text"],
            "step_count": self.step_count
        }

    def step(self, action):
        self.step_count += 1
        text = action["content"]["text"].lower()

        score = 0.0

        #  EASY TASK (classification)
        if self.current_task["type"] == "classification":
            if "delivery" in text:
                score += 0.5
            if "high" in text:
                score += 0.5

        #  MEDIUM TASK (response)
        elif self.current_task["type"] == "response":
            if "sorry" in text:
                score += 0.3
            if "refund" in text:
                score += 0.4
            if "delay" in text or "late" in text:
                score += 0.3

        #  HARD TASK (resolution)
        elif self.current_task["type"] == "resolution":
            if "refund" in text:
                score += 0.4
            if "escalate" in text:
                score += 0.3
            if "double" in text or "twice" in text:
                score += 0.3

        #  penalty for too many steps
        if self.step_count > 3:
            score -= 0.1

        # clamp score
        score = round(max(0.0, min(score, 1.0)), 2)

        done = score >= 0.9 or self.step_count >= 5

        reward = {
            "value": score,
            "feedback": "graded based on keywords"
        }

        return (
            {
                "task_id": self.current_task["id"],
                "input_text": self.current_task["text"],
                "step_count": self.step_count
            },
            reward,
            done,
            {}
        )

    def state(self):
        return {
            "task": self.current_task,
            "steps": self.step_count
        }
