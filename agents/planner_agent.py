import random
from agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent", 0.03)

    def act(self, query):
        if random.random() < 0.03:
            raise RuntimeError("Planning failed")
        return {"output": f"Plan for: {query}", "confidence": 0.80}
