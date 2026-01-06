import numpy as np

class LinUCB:
    def __init__(self, n_arms, dim, alpha=1.0):
        self.alpha = alpha
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select(self, x):
        scores = []
        for i in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            exploit = theta @ x
            explore = self.alpha * (x.T @ A_inv @ x) ** 0.5
            scores.append(float(exploit + explore))
        return int(max(range(len(scores)), key=lambda i: scores[i])), scores

    def update(self, arm, x, reward):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
