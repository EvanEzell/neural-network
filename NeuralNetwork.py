import numpy, sys 

# activation functions and their derivatives
def logistic(x) : return 1 / (1 + numpy.exp(-x))
def logistic_deriv(x) : return x * (1 - x)
def linear(x) : return x
def linear_deriv(x) : return 1

# loss functions and their derivatives
def square_error(prediction, target):
    total = 0
    for i in range(len(prediction)):
        total += .5 * ((prediction[i]-target[i])**2)
    return total
def log_loss(prediction, target):
    total = 0
    for i in range(len(prediction)):
        if target == 1:
            total += -log(prediction)
        else:
            total += -log(1 - prediction)
    return total
def square_error_deriv(prediction, target):
    return prediction - target
def log_loss_deriv(prediction, target):
    return - (target/prediction) + ((1-target)/(1-prediction))

def get_deriv(function):
    if function == logistic: return logistic_deriv 
    elif function == linear: return linear_deriv
    elif function == square_error: return square_error_deriv
    elif function == log_loss: return log_loss_deriv
    else: return None


class Neuron:
    
    def __init__(self, activation, num_inputs, eta, weights):
        self.activation = activation
        self.num_inputs = num_inputs
        self.eta = eta
        self.weights = []
        if weights == "random":
            for i in range(num_inputs+1):
               self.weights.append(numpy.random.random())
        else:
            self.weights = weights

    def activate(self, x):
        return self.activation(x)

    def calculate(self, data):
        total = 0;
        for i in range(len(data)):
            total += data[i] * self.weights[i]

        self.prev = data
        self.net = total + self.weights[len(data)]
        self.out = self.activate(self.net)

        return self.out

    def train(self, deriv):
        self.delta = get_deriv(self.activation)(self.out)

        weight_deltas = []
        for i in range(len(self.prev)):
            weight_deltas.append(deriv * self.prev[i])
            self.weights[i] -= self.eta * weight_deltas[i]

        self.weights[i+1] -= self.eta * deriv

        return weight_deltas 

    def print_neuron(self):
        print("num_inputs: " + str(self.num_inputs))
        print("eta: " + str(self.eta))
        count = 0
        for i in self.weights:
            print("weight[" + str(count) + "]: " + str(i))
            count += 1

class FullyConnectedLayer:
    def __init__(self, num_neurons, activation, num_inputs, eta, weights):
        self.activation = activation
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.eta = eta

        self.neurons = []
        if weights == "random":
            for i in range(self.num_neurons):
                self.neurons.append(Neuron(self.activation, self.num_inputs, self.eta, "random"))
        else:
            for i in range(self.num_neurons):
                self.neurons.append(Neuron(self.activation, self.num_inputs, self.eta, weights[i]))

    def calculate(self, data):
        output = []
        for i in range(self.num_neurons):
            output.append(self.neurons[i].calculate(data))

        return output

    def train(self, deriv):
        delta_sums = []
        for i in range(self.num_neurons):
            delta_sums.append(sum(self.neurons[i].train(deriv[i])))

        return(delta_sums)

    def print_layer(self):
        for i in range(self.num_neurons):
            print("neuron " + str(i))
            self.neurons[i].print_neuron()
            print("")

class NeuralNetwork:
    def __init__(self, num_layers, num_neurons, activation, num_inputs, loss, eta, weights):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.loss = loss
        self.eta = eta

        self.activation = []
        for i in range(num_layers):
            if activation == "logistic":
                self.activation.append(logistic)
            elif activation == "linear":
                self.activation.append(linear)

        self.layers = []
        prev_inputs = self.num_inputs
        if weights == "random":
            for i in range(self.num_layers):
                self.layers.append(FullyConnectedLayer(self.num_neurons[i], self.activation[i], prev_inputs, self.eta, "random"))
                prev_inputs = self.num_neurons[i]
        else:
            for i in range(self.num_layers):
                self.layers.append(FullyConnectedLayer(self.num_neurons[i], self.activation[i], prev_inputs, self.eta, weights[i]))
                prev_inputs = self.num_neurons[i]

    def calculate(self, data):
        output = data
        for layer in self.layers:
            output = layer.calculate(output)
        return output

    def calculateloss(self, prediction, target):
        return self.loss(prediction, target)

    def train(self, data, target):
        prediction = self.calculate(data)

        derivs = []
        for i in range(len(prediction)):
            derivs.append(get_deriv(self.loss)(prediction[i],target[i]) * get_deriv(self.activation[self.num_layers-1])(prediction[i]))

        for i in range(self.num_layers-2,-1,-1):
            derivs = self.layers[self.num_layers-1].train(derivs)

    def print_nn(self):
        for i in range(self.num_layers):
            print("Layer " + str(i))
            print("Number of Neurons in Layer: " + str(self.layers[i].num_neurons))
            self.layers[i].print_layer()

def main():
    choices = ['example', 'and', 'xor']
    if len(sys.argv) != 2 or sys.argv[1] not in choices:
        print("Please provide one of the following command line arguments:")
        print("'example', 'and', or 'xor'")
        exit()

    print("Choice: " + sys.argv[1])

    if sys.argv[1] == 'example':
        print("running example")
        num_inputs = 2
        num_layers = 2
        num_neurons = [2,2]
        weights = [[[.15,.20,.35],[.25,.30,.35]],[[.40,.45,.60],[.50,.55,.60]]]

        nn = NeuralNetwork(num_layers, num_neurons, "logistic", num_inputs, square_error, .5, weights)
        nn.train([.05,.10],[.01,.99])

        print("loss after")
        print(nn.calculateloss(nn.calculate([.05,.10]),[.01,.99]))
        
        nn.print_nn()

    elif sys.argv[1] == 'and':
        print("running and")
        num_inputs = 2
        num_layers = 2
        num_neurons = [1,1]
        weights = "random"

        inputs = [([0, 0], [0]), ([0, 1], [0]), ([1, 0], [0]), ([1, 1], [1])]
        nn = NeuralNetwork(num_layers, num_neurons, "logistic", num_inputs, square_error, .1, weights)

        for i in range(250000):
            for sample, target in inputs:
                nn.train(sample, target)

        print(nn.calculate([0,0]))
        print(nn.calculate([1,0]))
        print(nn.calculate([0,1]))
        print(nn.calculate([1,1]))
        
    else:
        print("running xor")
        num_inputs = 2
        num_layers = 2
        num_neurons = [4,1]
        weights = "random"

        inputs = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        nn = NeuralNetwork(num_layers, num_neurons, "logistic", num_inputs, square_error, .1, weights)

        for i in range(500000):
            for sample, target in inputs:
                nn.train(sample, target)

        print(nn.calculate([0,0]))
        print(nn.calculate([1,0]))
        print(nn.calculate([0,1]))
        print(nn.calculate([1,1]))

if __name__ == '__main__':
    main()

