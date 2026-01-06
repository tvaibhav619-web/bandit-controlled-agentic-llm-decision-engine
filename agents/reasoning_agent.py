import random
from agents.base_agent import BaseAgent

class ReasoningAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReasoningAgent", 0.02)

    def act(self, query):
        if random.random() < 0.05:
            raise RuntimeError("Reasoning failed")
        return {"output": f"Reasoned answer for: {query}", "confidence": 0.85}
