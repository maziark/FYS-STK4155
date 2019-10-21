import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def activation_function(values):
    return 1.0 / (1 + np.exp(values))


def der_activation_function(values):
    return values * (1 - values)


def MSE(x, x_):
    """
        Calculating the Mean Square Error.
        Argument (numpy array x, and \tilde{x})

        here x_ can either an array, or a constant.

        returns a double
    """
    return np.average((x - x_) ** 2)


class Layer:
    def __init__(self, input_size, node_size, weights_initial=None, bias=None):
        self.next = None
        self.error = None
        self.delta = None
        self.number_of_inputs = input_size
        self.number_of_nodes = node_size
        self.betas = weights_initial
        if weights_initial is None:
            self.betas = np.random.rand(input_size, node_size)

        self.bias = bias
        if bias is None:
            self.bias = np.random.rand(node_size)

    def forward(self, values):
        self.next = activation_function(
            np.dot(values, self.betas) + self.bias)
        return self.next

    def backward(self, values):
        return der_activation_function(values)


class NeuralNetwork:
    def __init__(self, hidden_layers, learning_rate, epsilon=0):
        self.epsilon = epsilon
        self.layers = []
        self.eta = learning_rate

        self.design(hidden_layers)

    def design(self, hidden_layers):
        for i in range(1, len(hidden_layers)):
            self.layers.append(Layer(hidden_layers[i - 1], hidden_layers[i]))

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

    def iterate(self, X, Y, max_iter=1000):
        mse = []
        for i in range(max_iter):
            for j in range(len(X)):
                self.backpropagation(np.array(X[j]), np.array(Y[j]))
                pred = self.layers[-1].next
            mse.append(np.mean(np.square(np.array(Y[j]) - pred)))

            if i > 100 and abs(mse[-1] - mse[-2]) <= self.epsilon:
                break
        return mse

    def predict(self, X):
        pred = self.forward(X)
        result_index = 0
        for i in range(len(pred)):
            if pred[i] > pred[result_index]:
                result_index = i

        return result_index


'''X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[1, 0], [1, 0], [1, 0], [0, 1]]
mlp = NeuralNetwork([2, 3, 2], 0.001)
mse = mlp.iterate(X, Y)



plt.plot(mse)
plt.title('Changes in MSE')
plt.xlabel('Epoch (every 10th)')
plt.ylabel('MSE')
plt.show()'''
