import time
import mlflow

from engine.embeddings import embed
from engine.reward import compute_reward
from engine.explain import explain
from utils.parallel import run_parallel
from utils.regret import RegretTracker


# ---------------------------------------------------
# Helper: detect planning / design type queries
# ---------------------------------------------------
def is_planning_query(query: str) -> bool:
    keywords = [
        "plan",
        "step",
        "steps",
        "design",
        "architecture",
        "migrate",
        "roadmap",
        "strategy",
        "system",
        "pipeline"
    ]
    q = query.lower()
    return any(k in q for k in keywords)


# ---------------------------------------------------
# Decision Controller
# ---------------------------------------------------
class DecisionController:
    def __init__(self, bandit, agents, chains):
        """
        bandit  : LinUCB or ThompsonSampling
        agents  : dict[name -> agent instance]
        chains  : list[list[str]] (bandit arms)
        """
        self.bandit = bandit
        self.agents = agents
        self.chains = chains
        self.regret_tracker = RegretTracker()

    def handle(self, query: str) -> dict:
        # ---------------------------------------------------
        # 1. Context (LLM embeddings)
        # ---------------------------------------------------
        context = embed(query)

        # ---------------------------------------------------
        # 2. Bandit selects agent chain
        # ---------------------------------------------------
        arm, scores = self.bandit.select(context)
        selected_chain = self.chains[arm]
        selected_agents = [self.agents[name] for name in selected_chain]

        # ---------------------------------------------------
        # 3. Execute agents (parallel)
        # ---------------------------------------------------
        start = time.time()
        failure = 0

        try:
            outputs = run_parallel(selected_agents, query)
        except Exception:
            outputs = []
            failure = 1

        latency = time.time() - start

        # ---------------------------------------------------
        # 4. Signals
        # ---------------------------------------------------
        cost = sum(a.cost for a in selected_agents)

        confidence = (
            sum(o.get("confidence", 0.0) for o in outputs)
            / max(1, len(outputs))
        )

        success = 1 if failure == 0 else 0

        # ---------------------------------------------------
        # 5. Base reward (multi-objective)
        # ---------------------------------------------------
        reward = compute_reward(
            success=success,
            cost=cost,
            latency=latency,
            confidence=confidence,
            failure=failure
        )

        # ---------------------------------------------------
        # 6. Reward shaping for planning tasks
        # ---------------------------------------------------
        under_planning_penalty = 0.0
        planning_bonus = 0.0

        if is_planning_query(query):
            # Penalize single-agent answers
            if len(selected_agents) == 1:
                under_planning_penalty = 0.4
                reward -= under_planning_penalty

            # Reward multi-agent collaboration
            if len(selected_agents) > 1:
                planning_bonus = 0.3
                reward += planning_bonus

        # ---------------------------------------------------
        # 7. Update bandit
        # ---------------------------------------------------
        self.bandit.update(arm, context, reward)

        # ---------------------------------------------------
        # 8. Regret (clamped)
        # ---------------------------------------------------
        optimal_estimate = max(scores)
        regret = max(0.0, optimal_estimate - reward)
        self.regret_tracker.update(optimal_estimate, reward)

        # ---------------------------------------------------
        # 9. MLflow logging
        # ---------------------------------------------------
        with mlflow.start_run():
            mlflow.log_metric("reward", reward)
            mlflow.log_metric("regret", regret)
            mlflow.log_metric("cost", cost)
            mlflow.log_metric("latency", latency)
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("failure", failure)

            mlflow.log_metric(
                "under_planning_penalty_applied",
                1 if under_planning_penalty > 0 else 0
            )
            mlflow.log_metric(
                "planning_bonus_applied",
                1 if planning_bonus > 0 else 0
            )

            mlflow.log_param("selected_chain", " -> ".join(selected_chain))
            mlflow.log_param("is_planning_query", is_planning_query(query))

        # ---------------------------------------------------
        # 10. Explainable response
        # ---------------------------------------------------
        return {
            "selected_chain": selected_chain,
            "outputs": outputs,
            "reward": reward,
            "regret": regret,
            "policy_explanation": explain(scores, self.chains)
        }
