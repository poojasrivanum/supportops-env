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