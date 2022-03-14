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

# Fungsi membaca input file
def _input(filename: str, with_input = False) -> tuple((FFNN, np.array, np.array)):
    f = open(filename, "r")
    f = f.readlines()
    f = [line.strip() for line in f]

    nLayer = int(f[0])
    f = f[1:]
    n_layer_neurons = []
    struct_model = {}

    for i in range(nLayer-1):
        struct_model[i] = {}
        n_layer_neurons.append(int(f[0]))
        struct_model[i]["b"] = [float(b) for b in f[1].split()]
        struct_model[i]["w"] = [[float(w) for w in weights.split()] for weights in f[2:(2 + int(f[0]))]]
        struct_model[i]["f"] = f[2 + int(f[0])]
        f = f[2 + int(f[0]) + 1:]

    n_layer_neurons.append(int(f[0]))
    
    if (with_input):
        n_input = int(f[1])
        f = f[2:]
        input_data = []
        for i in range(n_input):
            input = [int(x) for x in (f[i].split())]
            input_data.append(input)

        f = f[n_input:]
        validation_data = []
        for i in range(n_input):
            result = [int(y) for y in (f[i].split())]
            validation_data.append(result)

    model_layers = []
    for i in range (nLayer-1):
        weight = struct_model[i]["w"]
        weight.append(struct_model[i]["b"])
        layer = Layer(n_layer_neurons[i+1], np.array(weight), struct_model[i]["f"].lower())
        model_layers.append(layer)
    
    if (with_input):
        return FFNN(model_layers, np.array(input_data)), input_data, validation_data
    else:
        return FFNN(model_layers)

# Memperlihatkan koefisien dan struktur dari model
def showModel(model: FFNN): #masukan berupa FFNN
    initLayers = model.get_structure()

    countInput = len(initLayers[0])
    countLayer = len(initLayers[1])
    if(initLayers[0].ndim == 1):
        countInput = 1
    else:
        countInput = len(initLayers[0])
    for i in range(0, countInput):
        if(countInput == 1):
            inputLayer = initLayers[0]
    else:
        inputLayer = initLayers[0][i]
    output = inputLayer
    print("==============Data %d==============\n"%(i+1))
    print("-------------Input------------")
    print("Input Layer: ", inputLayer)
    print("------------------------------")
    for j in range(0, countLayer):
        weight = initLayers[1][j][1]
        bias = initLayers[1][j][2]
        activation = initLayers[1][j][3]
        combinedArr = initLayers[1][j][4]

        if (j == (countLayer - 1)):
            print("------ Output Layer ------" )
            print("Input: ", output)
            print("Weight: " , weight)
            print("Bias: " , bias)
            print('\n')
            output = activation_functions[activation](np.dot(combinedArr.transpose(), np.append(output, 1)))
            print("------------------------------")
            print(inputLayer)
            print("Predicted Result: ", model.forward(inputLayer))
        else:
            print("--- Hidden Layer %d ---" %(j+1))
            print("Input: ", output)
            print("H%d Weight: " %(j+1), weight)
            print("H%d Bias: " %(j+1), bias)
            print('\n')
            output = activation_functions[activation](np.dot(combinedArr.transpose(), np.append(output, 1)))
            print("------------------------------")

# Fungsi menghitung akurasi dari model
def calculate_accuracy(model: FFNN, input_set, validation_set: list, is_softmax = False):
    # returns range from 1..100 (percentage)
    predicted_set = model.predict(np.array(input_set))
    if (not isinstance(predicted_set, (list, np.ndarray))): return int(predicted_set == validation_set[0]) * 100
    if (len(predicted_set) != len(validation_set)): return None
    num_correct = 0
    for i in range(len(predicted_set)):
        if is_softmax:
            if (np.argmax(predicted_set[i]) == validation_set[i]): num_correct += 1
    else:
        if predicted_set[i] == validation_set[i]: num_correct += 1
    return num_correct / len(validation_set) * 100

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