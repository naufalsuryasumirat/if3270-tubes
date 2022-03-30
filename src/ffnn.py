import math
import numpy as np

# Activation functions
## Linear
linear = lambda x: x
linear = np.vectorize(linear)
## Sigmoid
sigmoid = lambda x: 1 / (1 + math.exp(-x))
sigmoid = np.vectorize(sigmoid)
## ReLU
relu = lambda x: float(max(0, x))
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

# Loss functions
## Linear, Sigmoid, ReLU
def general_loss(predict: np.array or int, target: list or int):
    if (isinstance(predict, type(int))): return 0.5 * ((target - predict) ** 2)
    sum = 0
    for i in range(len(target)):
        sum += ((target[i] - predict[i]) ** 2)
    return 0.5 * sum
## Softmax
def softmax_loss(predict: np.array, target: int):
    return -math.log(predict[target]) # base e
## Dict
loss_functions = {
    'linear': general_loss,
    'sigmoid': general_loss,
    'relu': general_loss,
    'softmax': softmax_loss
}

# Back-propagation functions (derivatives)
## Linear
linear_backprop = lambda x: 1
## Sigmoid
# sigmoid_backprop = lambda x: sigmoid(x) * (1 - sigmoid(x)) # or x * (1 - x)?
sigmoid_backprop = lambda x: x * (1 - x)
## ReLU
relu_backprop = lambda x: float(x >= 0)
relu_backprop = np.vectorize(relu_backprop)
## Softmax
def softmax_backprop(arr, targ):
    arr_copy = np.copy(arr)
    arr_copy[targ] = -(1 - arr_copy[targ])
    return arr_copy
## Dict
backprop_functions = {
    'linear': linear_backprop,
    'sigmoid': sigmoid_backprop,
    'relu': relu_backprop,
    'softmax': softmax_backprop
}

class ConfusionMatrix:
    def __init__(self, pred, val): # input has to be 2d array [numpy]
        if (len(pred) != len(val)): raise ValueError('pred and val length is not the same!')
        if (len(pred[0]) != len(val[0])): raise ValueError('number of class mismatch!')
        self.pred = pred
        self.val = val
        self.count = len(val) # all data count
        self.num_classes = len(val[0]) # number of classes
        self.num_per_class = [0] * self.num_classes # data count per class

        # col: predicted values, row: actual values [Confusion Matrix]
        self.matrix = self.generate_matrix(pred, val)

        self.true_positives = np.diag(self.matrix, k = 0) # True positives per class
        self.false_positives = self.calculate_false_positives() # False positives per class
        self.false_negatives = self.calculate_false_negatives() # False negatives per class
        self.true_negatives = self.calculate_true_negatives() # True negatives per class

        # Statistics per class
        self.accuracies = self.true_positives / self.num_per_class # Accuracy per class
        self.precisions = self.calculate_precisions() # Precision per class
        self.recalls = self.calculate_recalls() # Recall per class
        self.f1_scores = self.calculate_f1_scores() # F1_Score per class

        # Statistics for all
        self.accuracy = np.sum(self.true_positives) / self.count # Accuracy for all
        self.precision = np.average(self.precisions) # Precision for all
        self.recall = np.average(self.recalls) # Recall for all
        self.f1_score = np.average(self.f1_scores) # F1_Score for all

    def generate_matrix(self, pred, val):
        if (len(pred) != len(val)): return None
        matrix = np.zeros(shape=(len(val[0]), len(val[0])))
        for i in range(len(val)):
            matrix[np.argmax(val[i])][np.argmax(pred[i])] += 1
            self.num_per_class[np.argmax(val[i])] += 1
        return matrix
    
    def calculate_false_positives(self):
        arr = []
        for i in range(self.num_classes):
            col = self.matrix[:,i]
            arr.append(np.sum(col) - col[i])
        return np.array(arr)

    def calculate_false_negatives(self):
        arr = []
        for i in range(self.num_classes):
            row = self.matrix[i,:]
            arr.append(np.sum(row) - row[i])
        return np.array(arr)

    def calculate_true_negatives(self):
        arr = []
        for i in range(self.num_classes):
            mat = np.delete(np.delete(self.matrix, i, axis = 1), i, axis = 0)
            arr.append(np.sum(mat))
        return np.array(arr)

    def calculate_precisions(self):
        # true positive / (true positive + false positive)
        return self.true_positives / (self.true_positives + self.false_positives)

    def calculate_recalls(self):
        # true positive / (true positive + false negative)
        return self.true_positives / (self.true_positives + self.false_negatives)

    def calculate_f1_scores(self):
        # 2 * (precision * recall) / (precision + recall)
        return 2 * (self.precisions * self.recalls) / (self.precisions + self.recalls)

class Layer:
    # n_neuron: number of neuron, weights: weight matrix, activation: activation function
    def __init__(self, n_neuron: int, weights: np.array, activation: str) -> None:
        self.n_neuron = n_neuron # visualization purposes
        self.weights = weights # weights (including bias)
        self.activation = activation # activation type [linear, sigmoid, relu, softmax]
        self.act_function = activation_functions[activation] # activation function used
        self.loss_function = loss_functions[activation] # loss function used
        self.backprop_functions = backprop_functions[activation] # derivative activation functions
        self.result = [] # retain result of feed forward iteration
        self.deltas = np.zeros_like(self.weights) # initialize delta for backprop
        # every feed forward add result, every backprop add to delta

    def calculate(self, in_matrix: np.array) -> np.array:
        self.result = self.act_function(np.dot(self.weights.transpose(), in_matrix))
        return self.result # [a0, a1, a2, ..., an]

    def calculate_loss(self, prediction: (np.array or int), target: (list or int)) -> float: # used for calculating loss for output layer
        if (self.activation == "softmax"): return self.loss_function(prediction, np.argmax(target))
        return self.loss_function(prediction, target)
    
    def update_weight(self):
        self.weights += self.deltas # adding deltas to weights
        self.deltas = np.zeros_like(self.deltas) # resettings deltas for next mini-batch
        return self.weights # for verbose purpose
    
    def add_deltas(self, delta_matrix: np.array) -> None:
        self.deltas += delta_matrix
        return self.deltas # for verbose purpose
    
    def get_structure(self) -> tuple((int, np.array, np.array)):
        # n_neuron: int, weight matrix: np.array, bias weight matrix: np.array
        n_neuron = self.n_neuron
        weight_neuron = self.weights[:-1,]
        weight_bias = self.weights[-1:,].flatten()
        return (n_neuron, weight_neuron, weight_bias)

class FFNN:
    def __init__(self,  
            hidden_layers: list,
            input_layer = None,
            threshold = 0.5,
            learning_rate = 0.01,
            err_threshold = 0.001,
            max_iter = 5000,
            batch_size = 1) -> None:
        self.hidden_layers = hidden_layers
        self.output_layer = hidden_layers[-1]
        self.output_activation = self.output_layer.activation
        self.input_layer = input_layer
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.err_threshold = err_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size # default incremental SGD
    
    @staticmethod
    def generate_model(input_size: int, n_neurons: list, activations: list):
        if (len(n_neurons) != len(activations)): return None
        arr = []
        for i in range(len(n_neurons)):
            if (i == 0): arr.append(Layer(
                n_neurons[i], np.random.uniform(low = -1.0, high = 1.0, 
                    size = (input_size + 1, n_neurons[i])),
                    activations[i])
                )
            else: arr.append(Layer(
                n_neurons[i], np.random.uniform(low = -1.0, high = 1.0,
                    size = (n_neurons[i - 1] + 1, n_neurons[i])),
                    activations[i])
                )
        return FFNN(arr)

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
        if (self.output_layer.activation == 'softmax'): return output # usually used for multiclass
        if (self.output_layer.n_neuron > 1): return np.where(output > self.threshold, 1, 0) # multiclass non-softmax
        return int(output > self.threshold) # binary
    
    def fit(self, x_train, y_train, randomize = False, learning_rate = None, 
            batch_size = None, max_iter = None, 
            err_threshold = None, update_every = 250) -> None:
        if learning_rate is not None: self.learning_rate = learning_rate
        if batch_size is not None: self.batch_size = batch_size
        if max_iter is not None: self.max_iter = max_iter
        if err_threshold is not None: self.err_threshold = err_threshold
        for epoch in range(self.max_iter):
            training_data = x_train
            training_target = y_train
            if randomize:
                pass # randomize dataset x_train here
            # randomize dataset, for in range dataset do forward then backprop
            # if i + 1 % batch_size == 0 update_weight
            error_sum = 0 # initialize error (for comparing with err_threshold)
            for iter in range(len(y_train)):
                pred = self.predict(training_data[iter]) # results already encoded
                pred = self.output_layer.result # result before encoded
                error = self.output_layer.calculate_loss(pred, training_target[iter])
                self.backpropagate(training_data[iter], training_target[iter])
                error_sum += error
                if ((iter + 1) % self.batch_size == 0 or iter == len(training_target) - 1):
                    self.update_weights() # update weights (mini-batch)
            err_avg = error_sum / len(y_train)

            if (err_avg < self.err_threshold):
                break # stop fitting process when avg error < threshold

            if (epoch % update_every == 0):
                print("Epoch %d, Loss: %.6f" % (1 if epoch == 0 else epoch, err_avg))
        return
    
    def backpropagate(self, input, target): # update deltas for every layer
        err_term = 0
        for iter in reversed(range(0, len(self.hidden_layers))):
            prev_layer = None if iter == 0 else self.hidden_layers[iter - 1]
            prev_result = np.atleast_2d(np.append(input, 1)) if prev_layer == None \
                else np.atleast_2d(np.append(prev_layer.result, 1))
            if (iter == len(self.hidden_layers) - 1): # if output layer
                if (self.output_activation == "softmax"): # if softmax output layer
                    pred = self.output_layer.result
                    err_deriv = self.output_layer.backprop_functions(pred, np.argmax(target))
                    err_term = err_deriv
                    gradient = np.dot(prev_result.T,
                        np.atleast_2d(err_deriv))
                    delta = -self.learning_rate * gradient
                    self.output_layer.add_deltas(delta)
                    pass
                else: # if other output layer
                    pred = self.output_layer.result
                    err_deriv = -(np.array(target) - pred)
                    err_term = err_deriv
                    donet = self.output_layer.backprop_functions(pred)
                    gradient = np.dot(prev_result.T,
                        np.atleast_2d(err_deriv * donet))
                    delta = -self.learning_rate * gradient
                    self.output_layer.add_deltas(delta)
            else: # if hidden layer
                this_layer = self.hidden_layers[iter]
                next_layer = self.hidden_layers[iter + 1]
                err_term = np.add.reduce(next_layer.weights[:-1].T * 
                    np.atleast_2d(err_term).T, 0) / np.shape(err_term)[0]
                donet = this_layer.backprop_functions(this_layer.result) # no softmax in hidden layer
                gradient = np.dot(prev_result.T,
                    np.atleast_2d(err_term * donet)) # should be correct
                delta = -self.learning_rate * gradient
                self.hidden_layers[iter].add_deltas(delta)
                pass
        return
        
    def update_weights(self):
        for layer in self.hidden_layers:
            layer.update_weight()
        return
    
    def attach_input(self, input_layer: np.array) -> None:
        self.input_layer = input_layer
        return

    def attach_hidden_layer(self, hidden_layer: Layer) -> None:
        self.hidden_layers.append(hidden_layer)
        return

    def predict(self, input_layer: np.array) -> list: # input_layer without bias
        self.input_layer = input_layer
        return self.feed_forward()
    
    def predict_confusion(self, input_layer: np.array, validation_set: np.array) -> ConfusionMatrix:
        if len(input_layer) <= 1: return None, None # if not batch prediction
        self.input_layer = input_layer
        prediction = self.feed_forward()
        return ConfusionMatrix(prediction, validation_set) # can get prediction from confusion matrix

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
    x_train = np.array([
        [5.1, 3.5, 1.4, .2],
        [4.9, 3.0, 1.4, .2],
        [4.7, 3.2, 1.3, .2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8]
    ])

    y_train = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ])

    weight_input = np.random.uniform(low = -1.0, high = 1.0, size = (5, 4))
    weight_intermediate_1 = np.random.uniform(low = -1.0, high = 1.0, size = (5, 3))
    weight_intermediate_2 = np.random.uniform(low = -1.0, high = 1.0, size = (4, 4))
    weight_output = np.random.uniform(low = -1.0, high = 1.0, size = (5, 3))
    weight_output2 = np.random.uniform(low = -1.0, high = 1.0, size = (4, 3))

    layer_testfit_1 = Layer(4, weight_input, 'sigmoid')
    layer_testfit_2 = Layer(3, weight_intermediate_1, 'relu')
    layer_testfit_3 = Layer(5, weight_intermediate_2, 'linear')
    layer_testfit_4 = Layer(3, weight_output, 'softmax')
    ffnn_testfit = FFNN([layer_testfit_1, layer_testfit_2, layer_testfit_3, layer_testfit_4], batch_size = 2)
    ffnn_testfit = FFNN.generate_model(
        4, [4, 3, 5, 3], ['sigmoid', 'relu', 'linear', 'softmax']
    )

    print(ffnn_testfit.hidden_layers[1].weights)
    print("THIS IS TESTING")
    # ffnn_testfit.fit(x_train, y_train)
    print("FITTING ENDS HERE")
    print(ffnn_testfit.hidden_layers[1].weights)


    for data in x_train:
        print(ffnn_testfit.predict(data))
    
    layer_testfit2_1 = Layer(4, weight_input, 'sigmoid')
    layer_testfit2_2 = Layer(3, weight_intermediate_1, 'relu')
    layer_testfit2_3 = Layer(3, weight_output2, 'sigmoid')
    ffnn_testfit2 = FFNN([layer_testfit2_1, layer_testfit2_2, layer_testfit2_3])
    ffnn_testfit2.fit(x_train, y_train, batch_size=2)

    for data in x_train:
        print(ffnn_testfit2.predict(data))

    # print(softmax(np.dot(np.array([[0.75, 0.25, 0.11, 7], [0.75, 0.25, 0.11, 8]]), np.array([1, 1, 1, 1]).transpose())))