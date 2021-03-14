import numpy as np

class LRClassifier:
    def __init__(self, eta=0.01, max_iter=500):
        self.eta = eta
        self.max_iter = max_iter
        self.W = None

    def _linear_func(self, x):
        return np.dot(self.W[1:], x.T) + self.W[0]

    def _activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        cost = []
        self.W = np.zeros(X.shape[1] + 1)
        for i in range(self.max_iter):
            errors = self._activation(self._linear_func(X)) - y
            self.W[0] -= self.eta * errors.sum()
            self.W[1:] -= self.eta * np.dot(X.T, errors)
            cost.append((errors ** 2).sum() / 2.0)
        return self

    def predict(self, x):
        return np.where(self._activation(self._linear_func(x)) >= 0.5, 1, 0)

if __name__ == "__main__":
    pass