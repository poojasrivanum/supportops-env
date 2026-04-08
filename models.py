from pydantic import BaseModel
from typing import Dict, Any


class Observation(BaseModel):
    task_type: str
    content: Dict[str, Any]
    step_count: int


class Action(BaseModel):
    action_type: str
    payload: Dict[str, Any]


class Reward(BaseModel):
    score: float
    reason: str