import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


class LogisticRegression:
    def __init__(self, lr=0.1, max_iter=1000, epsilon=0):
        self.lr = lr
        self.max_iter = max_iter
        self.beta = None
        self.epsilon = epsilon
        self.cost_values = []

    def prob(self, x):
        return sigmoid(np.dot(x, self.beta))

    def cost_function(self, x, y):
        # We have to try to minimize this function
        y_pred = self.prob(x)
        log_y_pred = np.log(replaceZeroes(y_pred))
        loh_y_pred_ = np.log(replaceZeroes(1 - y_pred))
        return -1.0 * np.mean(y * log_y_pred + (1 - y) * loh_y_pred_)

    def gradient(self, x_train, y_train):
        # Calculates the gradient of decent
        return np.dot(x_train.T, (self.prob(x_train) - y_train)) / len(x_train)

    def train(self, x_train, y_train, x_valid, y_valid):
        self.beta = np.random.rand(x_train.shape[1], 1)
        for i in range(self.max_iter):
            self.cost_values.append(self.cost_function(x_train, y_train))

            change = self.lr * self.gradient(x_train, y_train)

            self.beta -= change

            # Early stopping if there isn't any significant improvement to be made
            if abs(np.mean(change)) < self.epsilon:
                break

    def predict(self, x):
        # [0, 0.5) -> 0
        # [0.5, 1) -> 1
        return self.prob(x) >= 0.5

    def confusion_matrix(self, x_valid, y_valid):
        conf_mat = [[0, 0], [0, 0]]
        y_pred = self.predict(x_valid)
        for i in range(len(y_pred)):
            conf_mat[int(y_pred[i][0])][y_valid[i][0]] += 1
        return conf_mat

    def accuracy(self, x_valid, y_valid):
        return np.sum(self.predict(x_valid) == y_valid) / len(y_valid)
