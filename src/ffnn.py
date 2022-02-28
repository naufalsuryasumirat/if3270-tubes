import math
import numpy as np

# Activation functions
## Linear
linear = lambda x: x
linear = np.vectorize(linear)
## Sigmoid (add threshold?)
sigmoid = lambda x: 1 / (1 + math.exp(-x))
sigmoid = np.vectorize(sigmoid)
## ReLU
relu = lambda x: max(0, x)
relu = np.vectorize(relu)
## Softmax
softmax = lambda x: np.exp(x) / np.exp(x).sum() # already used for vectors
## Dict
activation_functions = {
    'linear': linear,
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax
}

class FFNN:
    def __init__(self, input_layer: np.array, hidden_layers: list) -> None:
        self.input_layer = input_layer # adding bias
        # print(self.input_layer)
        self.hidden_layers = hidden_layers # last layer = output layer?
        # add batch feed forward?

    def feed_forward(self):
        output = self.input_layer # adding bias
        for i in range(0, len(self.hidden_layers)):
            output = self.hidden_layers[i].calculate(np.append(output, 1)) # adding bias
            # TODO add case for output layer? or change class attributes?
        return output

    def attach_hidden_layer(self, hidden_layer):
        self.hidden_layers.append(hidden_layer)

class Layer:
    # n_neuron: number of neuron, weights: weight matrix, activation: activation function
    def __init__(self, n_neuron: int, weights: np.array, activation: str) -> None:
        self.n_neuron = n_neuron # visualization purposes
        self.weights = weights
        self.activation = activation
        self.act_function = activation_functions[activation]

    def calculate(self, in_matrix: np.array) -> np.array:
        return self.act_function(np.dot(self.weights.transpose(), in_matrix))

if __name__ == "__main__":
    weight_sigmoid1 = [
        [1, 1],
        [1, 1],
        [0, -1]
    ]

    weight_sigmoid2 = [
        [20, -20],
        [20, -20],
        [-10, 30]
    ]

    weight_sigmoid3 = [
        [20],
        [20],
        [-30]
    ]

    weight_relu = [
        [1],
        [-1],
        [0]
    ]

    weight_relu2 = [
        [1, 1],
        [1, 1],
        [0, -1]
    ]   # n_column (banyak kolom): banyak neuron + 1 bias,
        # n_row (banyak baris): banyak neuron sebelumnya + 1 bias
    layer_sigmoid = Layer(2, np.array(weight_sigmoid2), 'sigmoid') #
    layer_relu = Layer(1, np.array(weight_sigmoid3), 'sigmoid')

    input = np.array([[0, 0]])

    ffnn = FFNN(input, [layer_sigmoid, layer_relu])
    print(ffnn.feed_forward())