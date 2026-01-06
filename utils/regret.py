class RegretTracker:
    def __init__(self):
        self.cumulative = 0.0

    def update(self, optimal, actual):
        regret = optimal - actual
        self.cumulative += regret
        return regret
