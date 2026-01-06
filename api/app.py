from fastapi import FastAPI
from bandits.linucb import LinUCB
from engine.controller import DecisionController
from agents.reasoning_agent import ReasoningAgent
from agents.planner_agent import PlannerAgent
from agents.verifier_agent import VerifierAgent

app = FastAPI()

agents = {
    "ReasoningAgent": ReasoningAgent(),
    "PlannerAgent": PlannerAgent(),
    "VerifierAgent": VerifierAgent()
}

CHAINS = [
    ["ReasoningAgent"],
    ["PlannerAgent", "VerifierAgent"],
    ["ReasoningAgent", "VerifierAgent"]
]

bandit = LinUCB(n_arms=len(CHAINS), dim=384)
controller = DecisionController(bandit, agents, CHAINS)

@app.post("/query")
def query(q: str):
    return controller.handle(q)
