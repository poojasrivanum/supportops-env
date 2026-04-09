---
title: SupportOpsEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
---

# SupportOpsEnv


## Overview

SupportOpsEnv is a lightweight simulation of real-world customer support workflows.  
It is designed to evaluate how well an agent can handle support tickets across three levels of difficulty.

The environment follows OpenEnv specifications and uses deterministic keyword-based grading with partial rewards.

---

## Tasks

### 1. Easy — Ticket Classification

**Objective:**  
Classify a customer issue into the correct category and priority.

**Example Input:**
"My order has not arrived"

**Expected Output:**
- Category: delivery
- Priority: high

**Scoring:**
- +0.5 for correct category ("delivery")
- +0.5 for correct priority ("high")

---

### 2. Medium — Response Generation

**Objective:**  
Generate a customer support response for a delayed order complaint.

**Example Input:**
"I want a refund, it's delayed"

**Evaluation Criteria:**
- Apology ("sorry") → +0.3  
- Refund mention → +0.4  
- Delay acknowledgment → +0.3  

---

### 3. Hard — Multi-step Resolution

**Objective:**  
Resolve a billing issue involving a double charge.

**Example Input:**
"I was charged twice and I am angry"

**Evaluation Criteria:**
- Refund action → +0.4  
- Escalation → +0.3  
- Double charge recognition → +0.3  

---

## Action Space

```json
{
  "action_type": "respond",
  "content": {
    "text": "string"
  }
}

