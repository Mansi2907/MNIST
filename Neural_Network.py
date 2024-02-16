from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None
        
    @abstractmethod
    def forward(self, input):
        pass
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        pass


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)  # initialize weights
        self.biases = np.zeros(output_size)  # initialize biases

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weight_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0)

        self.weights = self.weights - learning_rate * weight_gradient
        self.biases = self.biases - learning_rate * bias_gradient

        return input_gradient

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        sigmoid_derivative = self.output * (1 - self.output)
        input_gradient = sigmoid_derivative * output_gradient

        return input_gradient

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        tanh_derivative = 1 - np.tanh(self.output) ** 2
        input_gradient = tanh_derivative * output_gradient

        return input_gradient
    
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        shifted_input = input - np.max(input)  
        exp_input = np.exp(shifted_input)
        self.output = exp_input / np.sum(exp_input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient  

        return input_gradient
    
class CrossEntropyLossLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input,target):
        self.input = input
        self.target = target
        num_samples = input.shape[0]
        self.output = -np.sum(target * np.log(input)) / num_samples  
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * (self.input - self.target)
        return input_gradient

class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        self.input = input
        
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient
        for layer in reversed(self.layers):
            input_gradient = layer.backward(input_gradient, learning_rate)
        return input_gradient

    def save_weights(self, filepath):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                weights.append(layer.weights)
        np.savez(filepath, *weights)

    def load_weights(self, filepath):
        weights = np.load(filepath)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layer.weights = weights["arr_" + str(i)]
