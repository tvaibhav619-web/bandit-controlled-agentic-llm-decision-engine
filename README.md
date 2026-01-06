# Bandit-Controlled Agentic LLM Decision Engine

An **online-learning decision system** that uses **contextual bandits** to dynamically select **multi-agent LLM chains** based on **cost, latency, confidence, and reward**.

This project focuses on **decision intelligence**, not hardcoded routing or heavy MLOps pipelines.

---

## 🧠 Core Idea

Each **agent chain** is treated as a **bandit arm**.

For every query:
1. Convert query to **LLM embeddings**
2. Select an agent chain via a **contextual bandit**
3. Execute agents (parallel when applicable)
4. Compute a **multi-objective reward**
5. Update policy **online**
6. Track reward and regret with **MLflow**

---

## 🤖 Agents & Chains

**Agents**
- ReasoningAgent
- PlannerAgent
- VerifierAgent

**Chains (Bandit Arms)**
ReasoningAgent

PlannerAgent → VerifierAgent

ReasoningAgent → VerifierAgent

yaml
Copy code

---

## 🎰 Bandit Algorithms
- LinUCB (contextual, confidence-aware)
- Thompson Sampling (stochastic exploration)

---

## 🏆 Reward & Learning
Optimizes for:
- Task success
- Confidence
- Cost & latency
- Failure penalties
- Planning quality (reward shaping)

Tracks **regret** to verify online learning and policy convergence.

---

## 🔍 Explainability
Each decision returns:
- Selected agent chain
- Expected reward per chain
- Final reward and regret

---

## 🌐 API (FastAPI)

**Endpoint**
POST /query

pgsql
Copy code

**Example Response**
```json
{
  "selected_chain": ["PlannerAgent", "VerifierAgent"],
  "reward": 2.7,
  "regret": 0.03
}
📊 Monitoring (MLflow)
Logs:

Reward

Regret

Cost

Latency

Confidence

Planning bonuses / penalties

▶️ Run
bash
Copy code
pip install -r requirements.txt
mlflow ui
uvicorn api.app:app --reload
Swagger UI: http://127.0.0.1:8000/docs
MLflow UI: http://127.0.0.1:5000

📁 Structure
css
Copy code
agents/    bandits/    engine/    utils/
api/       main.py    requirements.txt
