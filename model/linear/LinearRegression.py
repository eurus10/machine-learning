import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self, mode="lsm", eta=0.1, max_iter=100):
        _fitting_funcs = {"lsm":self._least_square, "gdm":self._gradient_descent}
        self.fitting_func = _fitting_funcs[mode]
        self.eta = eta
        self.max_iter = max_iter
        self.W = None

    def _linear_func(self, X):
        return X @ self.W[1:] + self.W[0] # z = w0 + w1 * x1 + w2 * x2... = W.T @ x

    def _least_square(self, X, y):
        X0 = np.ones((X.shape[0], 1))
        X = np.hstack([X0, X])
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y # W = (X.T @ X)^-1 @ X.T @ y

    def _cost_function(self, delta):
        return sum(delta ** 2) / delta.shape[0]

    def _gradient_descent(self, X, y):
        cost = []
        self.W = np.zeros(X.shape[1] + 1)
        for i in range(self.max_iter):
            delta = y - self._linear_func(X)
            self.W[0] += self.eta * sum(delta) / X.shape[0]
            self.W[1:] += self.eta * (X.T @ delta) / X.shape[0]
            cost.append(self._cost_function(delta))
        plt.figure()
        plt.plot(range(self.max_iter), cost, c="green")
        plt.show()

    def fit(self, X, y):
        self.fitting_func(X, y)
        return self

    def predict(self, X):
        return self._linear_func(X)

if __name__ == "__main__":
    pass
