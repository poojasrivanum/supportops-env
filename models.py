from pydantic import BaseModel
from typing import Dict

class Observation(BaseModel):
    task_id: str
    input_text: str
    step_count: int

class Action(BaseModel):
    action_type: str
    content: Dict[str, str]

class Reward(BaseModel):
    value: float
    feedback: str
