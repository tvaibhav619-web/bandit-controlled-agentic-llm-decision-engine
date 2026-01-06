def compute_reward(success, cost, latency, confidence, failure):
    return (
        2.0 * success +
        1.0 * confidence -
        1.5 * cost -
        0.01 * latency -
        2.0 * failure
    )
def planning_bonus(query):
    keywords = ["step", "plan", "design", "architecture", "migrate"]
    return 0.5 if any(k in query.lower() for k in keywords) else 0.0

