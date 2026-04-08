---
title: SupportOpsEnv
emoji: "🤖"
colorFrom: indigo
colorTo: purple
sdk: docker
---

## Overview

SupportOpsEnv is a real-world simulation environment for customer support workflows. It allows AI agents to perform tasks such as ticket classification, response generation, and issue resolution using the OpenEnv standard API.

The environment evaluates how effectively an agent can handle practical support tasks.

---

## Tasks

### Easy – Ticket Classification
Classify a support request into category and priority.

### Medium – Response Generation
Generate an appropriate response including apology and resolution.

### Hard – Issue Resolution
Handle complex issues such as refunds, escalation, and multiple-step reasoning.

---

## Observation Space

```json
{
  "task_id": "string",
  "input_text": "string",
  "step_count": "integer"
}

Action Space

{
"action_type": "respond",
"content": {
"text": "string"
}
}

Reward Function
Partial rewards based on keyword correctness
Higher score for better responses
Penalizes extra steps using decay
Score always between 0.0 and 1.0
Environment API
reset() returns initial observation
step(action) returns observation, reward, done, info
state() returns current state
Setup

Install dependencies:

pip install -r requirements.txt

Set environment variables:

API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_api_key

Run inference:

python inference.py

Docker

Build:

docker build -t supportops .

Run:

docker run supportops

Baseline Performance

Easy: ~0.8
Medium: ~0.7
Hard: ~0.6

Notes
Deterministic grading
No randomness in rewards
Designed for reproducibility
