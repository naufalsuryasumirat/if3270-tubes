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
softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=0) # already used for vectors
## Dict
activation_functions = {
    'linear': linear,
    'sigmoid': sigmoid,
    'relu': relu,
    'softmax': softmax
}

class Layer:
    # n_neuron: number of neuron, weights: weight matrix, activation: activation function
    def __init__(self, n_neuron: int, weights: np.array, activation: str) -> None:
        self.n_neuron = n_neuron # visualization purposes
        self.weights = weights
        self.activation = activation
        self.act_function = activation_functions[activation]

    def calculate(self, in_matrix: np.array) -> np.array:
        return self.act_function(np.dot(self.weights.transpose(), in_matrix))
    
    def get_structure(self) -> tuple((int, np.array, np.array)):
        # n_neuron: int, weight matrix: np.array, bias weight matrix: np.array
        n_neuron = self.n_neuron
        weight_neuron = self.weights[:-1,]
        weight_bias = self.weights[-1:,].flatten()
        return (n_neuron, weight_neuron, weight_bias)

class FFNN:
    def __init__(self,  hidden_layers: list, input_layer = None, threshold = 0.5) -> None:
        self.hidden_layers = hidden_layers
        self.output_layer = hidden_layers[-1]
        self.input_layer = input_layer
        self.threshold = threshold

    def feed_forward(self) -> (np.array or None):
        if (isinstance(self.input_layer, type(None))): return None
        if len(self.input_layer.shape) == 1: return self.forward(self.input_layer)
        else:
            outputs = []
            for data in self.input_layer: outputs.append(self.forward(data))
            if (self.output_layer.activation == 'softmax'): return outputs
            return np.array(outputs).flatten()
    
    def forward(self, input) -> (np.array or None):
        output = input
        for i in range(0, len(self.hidden_layers)):
            output = self.hidden_layers[i].calculate(np.append(output, 1))
        if (self.output_layer.activation == 'softmax'): return output
        return int(output > self.threshold)

    def attach_hidden_layer(self, hidden_layer: Layer) -> None:
        self.hidden_layers.append(hidden_layer)

    def predict(self, input_layer: np.array) -> list: # input_layer without bias
        self.input_layer = input_layer
        return self.feed_forward()

    def get_structure(self) -> tuple((np.array, list)):
        return (self.input_layer, [layer.get_structure() for layer in self.hidden_layers])

if __name__ == "__main__":
    # XOR Dataset
    input = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1] 
    ])

    # Sigmoid Model (PPT)
    weight_sigmoid_1 = np.array([
        [20, -20],
        [20, -20],
        [-10, 30] # weight bias
    ])

    weight_sigmoid_2 = np.array([
        [20],
        [20],
        [-30] # weight bias
    ])  # n_column (banyak kolom): banyak neuron + 1 bias,
        # n_row (banyak baris): banyak neuron sebelumnya + 1 bias

    # Relu Model (PPT)
    weight_relu_1 = np.array([
        [1, 1],
        [1, 1],
        [0, -1] # weight bias
    ])

    weight_relu_2 = np.array([ # Linear too
        [1],
        [-2],
        [0] # weight bias
    ])

    weight_softmax = np.array([
        [-10, 17],
        [-20, 18],
        [30, -10]
    ])

    # print(input[-1:,].flatten())
    # print(input[:-1,])

    # Input FFN
    layer_sigmoid = Layer(2, weight_sigmoid_1, 'sigmoid')
    layer_sigmoid_2 = Layer(1, weight_sigmoid_2, 'sigmoid')
    ffnn_sig = FFNN([layer_sigmoid, layer_sigmoid_2], input)

    # Model Relu
    layer_relu_2 = Layer(2, weight_relu_1, 'relu')
    layer_relu_3 = Layer(1, weight_relu_2, 'relu')
    ffnn_relu = FFNN([layer_relu_2, layer_relu_3], input)

    # Model Relu-Linear
    layer_relu_4 = Layer(2, weight_relu_1, 'relu')
    layer_linear = Layer(1, weight_relu_2, 'linear')
    ffnn_reli = FFNN([layer_relu_4, layer_linear], input)

    # Model Sigmoid-Softmax
    layer_sigmoid_3 = Layer(2, weight_sigmoid_1, 'sigmoid')
    layer_softmax = Layer(2, weight_softmax, 'softmax')
    ffnn_sigmax = FFNN([layer_sigmoid_3, layer_softmax], input)

    # Contoh get_structure untuk Layer
    print(layer_sigmoid.get_structure())

    # Contoh get_structure untuk FFNN
    print(ffnn_sig.get_structure())

    print(ffnn_sig.feed_forward())
    print(ffnn_relu.feed_forward())
    print(ffnn_reli.feed_forward())
    print(ffnn_sigmax.feed_forward())

    print(ffnn_reli.predict(np.array([0, 1])))

    print(softmax(np.dot(np.array([[0.75, 0.25, 0.11, 7], [0.75, 0.25, 0.11, 8]]), np.array([1, 1, 1, 1]).transpose())))