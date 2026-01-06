"""
Local CLI runner for Bandit-Controlled Agentic LLM Decision Engine.
Used for quick testing and debugging (non-API mode).
"""

from bandits.linucb import LinUCB
from engine.controller import DecisionController
from agents.reasoning_agent import ReasoningAgent
from agents.planner_agent import PlannerAgent
from agents.verifier_agent import VerifierAgent

# -----------------------------
# Initialize agents
# -----------------------------
agents = {
    "ReasoningAgent": ReasoningAgent(),
    "PlannerAgent": PlannerAgent(),
    "VerifierAgent": VerifierAgent(),
}

# -----------------------------
# Define agent chains (bandit arms)
# -----------------------------
AGENT_CHAINS = [
    ["ReasoningAgent"],
    ["PlannerAgent", "VerifierAgent"],
    ["ReasoningAgent", "VerifierAgent"],
]

# -----------------------------
# Initialize bandit
# -----------------------------
bandit = LinUCB(
    n_arms=len(AGENT_CHAINS),
    dim=384,        # embedding size (MiniLM)
    alpha=1.0
)

# -----------------------------
# Initialize controller
# -----------------------------
controller = DecisionController(
    bandit=bandit,
    agents=agents,
    chains=AGENT_CHAINS
)

# -----------------------------
# Interactive loop
# -----------------------------
def run():
    print("\n🧠 Bandit-Controlled Agentic LLM Decision Engine")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User Query ➜ ")
        if query.lower() == "exit":
            break

        result = controller.handle(query)

        print("\n--- Decision Result ---")
        print("Selected Chain :", result["selected_chain"])
        print("Reward         :", round(result["reward"], 3))
        print("Regret         :", round(result["regret"], 3))
        print("Policy Scores  :", result["policy_explanation"])
        print("-----------------------\n")


if __name__ == "__main__":
    run()
