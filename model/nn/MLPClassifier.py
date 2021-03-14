import numpy as np

class MLPClassifier:
    def __init__(self, feature_n, hidden=None, label_n=2, eta=0.1, max_iter=100, activate_func="tanh"):
        # 函数字典
        funcs = {
            "sigmoid": (self._sigmoid, self._dsigmoid),
            "tanh": (self._tanh, self._dtanh),
            "relu": (self._relu, self._drelu)
        }
        # 接口参数
        self.feature_n = feature_n
        if not hidden:
            self.hidden = [10]
        else:
            self.hidden = hidden
        self.deep = len(self.hidden) + 1
        self.label_n = label_n
        self.eta = eta
        self.max_iter = max_iter
        self.activate_func, self.dacticate_func = funcs[activate_func]
        # 拟合缓存
        self.W = [] # 权重
        self.g = list(range(self.deep)) # 梯度
        self.v = [] # 神经元输出值
        # 初始化数据
        for d in range(self.deep):
            if d == 0:
                self.W.append(np.random.random([self.hidden[d], feature_n]))
            elif d == self.deep - 1:
                self.W.append(np.random.random([label_n, self.hidden[d - 1]]))
            else:
                self.W.append(np.random.random([self.hidden[d], self.hidden[d - 1]]))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _dsigmoid(self, h): # h = sigmoid(z)
        return h * (1 - h)

    def _tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def _dtanh(self, h): # h = tanh(z)
        return 1 - h ** 2

    def _relu(self, z):
        return (np.abs(z) + z) / 2

    def _drelu(self, h): # h = relu(z)
        return np.where(h > 0, 1, 0)

    def _linear_input(self, x, d):
        return x @ self.W[d].T

    def _forward_propagation(self, x): # 前向传播
        self.v.clear()
        value = None
        for d in range(self.deep):
            if d == 0:
                value = self.activate_func(self._linear_input(x, d))
            elif d == self.deep - 1:
                value = self._sigmoid(self._linear_input(self.v[d - 1], d))
            else:
                value = self.activate_func(self._linear_input(self.v[d - 1], d))
            self.v.append(value)
        return value

    def _back_propagation(self, y): # 反向传播
        for d in range(self.deep - 1, -1, -1):
            if d == self.deep - 1:
                self.g[d] = (y - self.v[d]) * self._dsigmoid(self.v[d])
            else:
                self.g[d] = self.g[d + 1] @ self.W[d + 1] * self.dacticate_func(self.v[d])

    def _bp(self, X, y):
        for i in range(self.max_iter):
            for x, yi in zip(X, y):
                self._forward_propagation(x) # 前向传播
                self._back_propagation(yi) # 反向传播
                # 更新权重
                for d in range(self.deep):
                    if d == 0:
                        self.W[d] += self.g[d].reshape(-1, 1) @ x.reshape(1, -1) * self.eta
                    else:
                        self.W[d] += self.g[d].reshape(-1, 1) @ self.v[d - 1].reshape(1, -1) * self.eta

    def _encoder(self, y):
        y_new = []
        for yi in y:
            yi_new = np.zeros(self.label_n)
            yi_new[yi] = 1
            y_new.append(yi_new)
        return y_new

    def fit(self, X, y):
        y = self._encoder(y)
        self._bp(X, y)
        return self

    def _predict(self, x):
        y_c = self._forward_propagation(x)
        return np.argmax(y_c)

    def predict(self, X):
        y = []
        for x in X:
            y.append(self._predict(x))
        return np.array(y)

if __name__ == "__main__":
    pass