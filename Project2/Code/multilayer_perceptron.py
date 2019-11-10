import numpy as np
from sklearn.utils import shuffle


def activation_sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def der_activation_sigmoid(values):
    return values * (1 - values)


def activation_linear(values):
    return values


def der_activation_linear(values):
    return values


def activation_tanh(values):
    return np.tanh(values)


def der_activation_tanh(values):
    return 1. - values ** 2


def MSE(x, x_):
    """
        Calculating the Mean Square Error.
        Argument (numpy array x, and \tilde{x})

        here x_ can either an array, or a constant.

        returns a double
    """
    return np.mean(np.square(x - x_))


class Layer:
    def __init__(self, input_size, node_size, activation='sigmoid'):
        # Statistics
        self.result = None
        self.error = None
        self.delta = None
        self.last_value = None

        # Features
        self.number_of_inputs = input_size
        self.number_of_nodes = node_size
        self.activation = activation

        # Values
        self.bias = np.random.rand(node_size)
        self.betas = np.random.rand(input_size, node_size)

    def forward(self, values):
        tmp = np.dot(values, self.betas) + self.bias
        self.result = self.activate(tmp)
        return self.result

    def activate(self, value):
        # sigmoid:
        if 'sig' in self.activation:
            return activation_sigmoid(value)

        # tanh
        elif 'tanh' in self.activation:
            return activation_tanh(value)

        # linear
        else:
            return value

    def backward(self, value):
        # sigmoid:
        if 'sig' in self.activation:
            return der_activation_sigmoid(value)

        # tanh
        elif 'tanh' in self.activation:
            return der_activation_tanh(value)

        # linear
        else:
            return value


class NeuralNetwork:
    def __init__(self, learning_rate=0.01, max_iter=100, epsilon=0):
        # design:
        self.mse_score = []
        self.layers = []

        # early stopping
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.eta = learning_rate

    def new_layer(self, design):
        self.layers.append(Layer(design['input_size'], design['number_of_nodes'], design['activation_function']))

    def forward(self, x_train):
        next_input = x_train
        for layer in self.layers:
            next_input = layer.forward(next_input)

        return next_input

    def predict(self, x):
        result = self.forward(x)

        if result.ndim == 1:
            return np.argmax(result)

        else:
            return np.argmax(result, axis=1)

    def backward(self, x_train, y_train):
        result = self.forward(x_train)
        number_layers = len(self.layers)
        for i in reversed(range(number_layers)):
            if self.layers[i] == self.layers[-1]:
                self.layers[i].error = y_train - result
                self.layers[i].delta = self.layers[i].error * self.layers[i].backward(result)

            else:
                # weighted error
                self.layers[i].error = np.dot(self.layers[i + 1].betas, self.layers[i + 1].delta)
                self.layers[i].delta = self.layers[i].error * self.layers[i].backward(self.layers[i].result)

        for i in range(number_layers):
            if i == 0:
                tmp_x = x_train
            else:
                tmp_x = self.layers[i - 1].result

            tmp_x = np.atleast_2d(tmp_x)
            self.layers[i].betas += self.layers[i].delta * tmp_x.T * self.eta

    def train(self, x_train, y_train):

        for i in range(self.max_iter):
            tmp_x_train, tmp_y_train = shuffle(x_train, y_train, random_state=0)
            n_train = len(tmp_x_train)
            n_valid = int(n_train / 5)
            x_valid, y_valid = tmp_x_train[:n_valid], tmp_y_train[:n_valid]
            for x in range(n_train):
                self.backward(tmp_x_train[x], tmp_y_train[x])

            self.mse_score.append(MSE(y_valid, self.forward(x_valid)))

            if i > 10 and abs(self.mse_score[-1] - self.mse_score[-2]) <= self.epsilon:
                break
