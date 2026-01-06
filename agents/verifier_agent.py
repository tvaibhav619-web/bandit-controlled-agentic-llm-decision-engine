import random
from agents.base_agent import BaseAgent

class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("VerifierAgent", 0.01)

    def act(self, query):
        if random.random() < 0.02:
            raise RuntimeError("Verification failed")
        return {"output": f"Verified result for: {query}", "confidence": 0.90}
