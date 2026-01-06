import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.success = np.ones(n_arms)
        self.failure = np.ones(n_arms)

    def select(self):
        samples = np.random.beta(self.success, self.failure)
        return int(samples.argmax()), samples.tolist()

    def update(self, arm, reward):
        if reward > 0:
            self.success[arm] += 1
        else:
            self.failure[arm] += 1
