import numpy as np
import matplotlib.pyplot as plt
import random


# np.random.seed(1340)


def activation_sigmoid(values):
    return 1.0 / (1 + np.exp(values))


def der_activation_sigmoid(values):
    return values * (1 - values)


def activation_linear(values):
    return values


def der_activation_linear(values):
    return values


def activation_tanh(values):
    return np.tanh(values)


def der_activation_tanh(values):
    return 1. / (np.cosh(values) ** 2)


def MSE(x, x_):
    """
        Calculating the Mean Square Error.
        Argument (numpy array x, and \tilde{x})

        here x_ can either an array, or a constant.

        returns a double
    """
    return np.average((x - x_) ** 2)


class Layer:
    def __init__(self, input_size, node_size, weights_initial=None, bias=None, activation='sigmoid'):
        self.next = None
        self.error = None
        self.delta = None
        self.number_of_inputs = input_size
        self.number_of_nodes = node_size
        self.betas = weights_initial
        self.activation = activation
        if weights_initial is None:
            self.betas = np.random.rand(input_size, node_size)

        self.bias = bias
        if bias is None:
            self.bias = np.random.rand(node_size)

    def forward(self, values):
        tmp = np.dot(values, self.betas) + self.bias
        self.next = eval('activation_{}({})'.format(self.activation, tmp))
        return self.next

    def backward(self, values):
        return eval('der_activation_{}({})'.format(self.activation, values))


class NeuralNetwork:
    def __init__(self, hidden_layers, learning_rate, epsilon=0):
        self.epsilon = epsilon
        self.layers = []
        self.hidden_layers = hidden_layers
        self.eta = learning_rate

    def run(self, X, Y, X_test, Y_test, iter=20):
        tmp_layers = []
        tmp_error = []

        for i in range(iter):
            self.layers = self.design(self.hidden_layers)
            tmp_error.append(self.iterate(X, Y, X_test, Y_test))
            tmp_layers.append(self.layers)
        self.layers = tmp_layers[np.array(tmp_error).argmin()]
        return tmp_error

    def design(self, hidden_layers):
        layers = []
        for i in range(1, len(hidden_layers)):
            layers.append(Layer(hidden_layers[i - 1], hidden_layers[i]))
        return layers

    def forward(self, values):
        for layer in self.layers:
            values = layer.forward(values)

        return values

    def backpropagation(self, X, Y):
        pred = self.forward(X)

        # First dealing with the last layer:
        tmp = self.layers[-1]
        tmp.error = Y - pred  # tmp.next
        tmp.delta = tmp.error * tmp.backward(tmp.next)

        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            current_layer.error = np.dot(next_layer.betas, next_layer.delta)
            current_layer.delta = current_layer.error * current_layer.backward(current_layer.next)

        # improve
        for layer in self.layers:
            if layer == self.layers[0]:
                current_input = np.atleast_2d(X)
            # print(self.eta, layer.delta.shape, current_input.T.shape, layer.betas.shape)
            layer.betas += layer.delta * current_input.T * self.eta
            current_input = np.atleast_2d(layer.next)

    def iterate(self, X, Y, X_test, Y_test, max_iter=1000):
        count_iter = 0
        error = [float('inf')]
        order = list(range(0, len(X)))
        while max_iter > count_iter:
            count_iter += 1
            random.shuffle(order)

            for j in order:
                self.backpropagation(np.array(X[j]), np.array(Y[j]))

            output = [self.forward(np.array(x)) for x in X_test]
            error.append(0.5 * sum([sum(output[i] - Y_test[i]) ** 2.0 for i in range(len(Y_test))]))
            if error[-1] > error[-2]:
                break

        return error

    def predict(self, X):
        single_flag = False
        if not isinstance(X, list):
            X = [X]
            single_flag = True

        result = []
        for x in X:
            pred = self.forward(np.array(x))
            result.append(np.argmax(pred))
        if single_flag:
            result = result[0]

        return result

    def confusion_matrix(self, X, Y):
        confused_matrix = np.zeros((Y.shape[1], Y.shape[1]))
        for i in range(len(X)):
            actual = np.argmax(Y[i])
            predicted = self.predict(np.array(X[i]))

            confused_matrix[predicted][actual] += 1

        print(confused_matrix)


'''X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[1, 0], [1, 0], [1, 0], [0, 1]]
mlp = NeuralNetwork([2, 3, 2], 0.001)
mse = mlp.iterate(X, Y)



plt.plot(mse)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()'''
