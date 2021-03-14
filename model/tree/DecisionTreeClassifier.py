import numpy as np
from scipy import stats as ss

class Node:
    def __init__(self, value=None, feature=None, y=None):
        self.value = value
        self.feature = feature
        self.y = y
        self.sub_node = []

    def append(self, node):
        self.sub_node.append(node)


class DecisionTreeClassifier:
    def __init__(self):
        self.root = Node()

    def _gini(self, y): # 基尼值
        y_ps = []
        y_unque = np.unique(y)
        for y_u in y_unque:
            y_ps.append(np.sum(y == y_u) / len(y))
        return 1 - sum(np.array(y_ps) ** 2)

    def _gini_index(self, X, y, feature): # 特征feature的基尼指数
        X_y = np.hstack([X, y.reshape(-1, 1)])
        unique_feature = np.unique(X_y[:, feature])
        gini_index = []
        for uf in unique_feature:
            sub_y = X_y[X_y[:, feature] == uf][:, X_y.shape[1] - 1]
            gini_index.append(len(sub_y) / len(y) * self._gini(sub_y))
        return sum(gini_index), feature

    def _best_feature(self, X, y, features): # 选择基尼指数最低的特征
        return min([self._gini_index(X, y, feature) for feature in features], key=lambda x:x[0])[1]

    def _post_pruning(self, X, y):
        nodes_mid = [] # 栈，存储所有中间结点
        nodes = [self.root] # 队列，用于辅助广度优先遍历
        while nodes: # 通过广度优先遍历找到所有中间结点
            node = nodes.pop(0)
            if node.sub_node:
                nodes_mid.append(node)
                for sub in node.sub_node:
                    nodes.append(sub)
        while nodes_mid: # 开始剪枝
            node = nodes_mid.pop(len(nodes_mid) - 1)
            y_pred = self.predict(X)
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y, y_pred)
            temp = node.sub_node
            node.sub_node = None
            if accuracy_score(y, self.predict(X)) <= score:
                node.sub_node = temp

    def fit(self, X, y):
        # 将数据集划分为训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3)
        queue = [[self.root, list(range(X_train.shape[0])), list(range(X_train.shape[1]))]]
        while queue: # 广度优先生成树
            node, indexs, features = queue.pop(0)
            node.y = ss.mode(y_train[indexs])[0][0] # 这里给每一个结点都添加了类标是为了防止测试集出现训练集中没有的特征值
            # 如果样本全部属于同一类别
            unique_y = np.unique(y_train[indexs])
            if len(unique_y) == 1:
                continue
            # 如果无法继续进行划分
            if len(features) < 2:
                if len(features) == 0 or len(np.unique(X_train[indexs, features[0]])) == 1:
                    continue
            # 选择最优划分特征
            feature = self._best_feature(X_train[indexs], y_train[indexs], features)
            node.feature = feature
            features.remove(feature)
            # 生成子节点
            for uf in np.unique(X_train[indexs, feature]):
                sub_node = Node(value=uf)
                node.append(sub_node)
                new_indexs = []
                for index in indexs:
                    if X_train[index, feature] == uf:
                        new_indexs.append(index)
                queue.append([sub_node, new_indexs, features])
        self._post_pruning(X_valid, y_valid)
        return self

    def _predict(self, X): # 单独处理每一个样本，这里使用广播不太合适
        node = self.root
        while node.sub_node:
            found = False
            for sub in node.sub_node:
                if X[node.feature] == sub.value:
                    node = sub
                    found = True
                    break
            if not found: # 训练集出现了测试集中没有出现过的特征值
                break
        return node.y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self._predict(X[i])
        return y_pred.astype(int)


if __name__ == "__main__":
    pass
