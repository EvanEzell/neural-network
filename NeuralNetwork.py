import numpy, sys 

debug = False

# activation functions and their derivatives
def logistic(x) : return 1 / (1 + numpy.exp(-x))
def logistic_deriv(x) : return logistic(x) * (1 - logistic(x))
def linear(x) : return x
def linear_deriv(x) : return 1

# loss functions and their derivatives
def square_error(prediction, target):
    total = 0
    for i in range(len(prediction)):
        total += (prediction[i]-target[i])**2
    return total
def mse(data):
    total = 0
    for sample in data:
        prediction, target = sample
        total += square_error(prediction,target)
    return total/len(data)
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

# get the derivative of a function
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
        if debug:
            print("in neuron train")
            print("deriv: " + str(deriv))
            print("net: " + str(self.net))
            print("deriv of net: " + str(get_deriv(self.activation)(self.net)))
        self.delta = get_deriv(self.activation)(self.net)*deriv

        if debug:
            print("node_delta: " + str(self.delta))
            print("previous: " + str(self.prev))
            print("old weights: " + str(self.weights))


        weight_deltas = []
        for i in range(len(self.prev)):
            weight_deltas.append(self.delta * self.weights[i])

        if debug: print("weight_delats: " + str(weight_deltas))

        # update weights
        for i in range(len(self.prev)):
            self.weights[i] -= self.eta * self.delta * self.prev[i]
        self.weights[i+1] -= self.eta * self.delta

        if debug: print("new_weights: " + str(self.weights))

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
                self.neurons.append(Neuron(self.activation,
                                           self.num_inputs,
                                           self.eta,
                                           "random"))
        else:
            for i in range(self.num_neurons):
                self.neurons.append(Neuron(self.activation,
                                           self.num_inputs,
                                           self.eta,
                                           weights[i]))

    def calculate(self, data):
        output = []
        for i in range(self.num_neurons):
            output.append(self.neurons[i].calculate(data))

        return output

    def train(self, derivs):
        if debug:
            print("in layer train")
            print("derivs: " + str(derivs))
        delta_sums = []
        if debug:
            print("num neurons in layer: " + str(self.num_neurons))
            print("training neuron 0")
        delta_sums = self.neurons[0].train(derivs[0])
        for i in range(1,self.num_neurons):
            if debug:
                print("training neuron " + str(i))
                print(self.neurons[i].train(derivs[i]))
            delta_sums = numpy.add(delta_sums,self.neurons[i].train(derivs[i]))

        if debug: print("delta_sums: " + str(delta_sums))
        return(delta_sums)

    def print_layer(self):
        for i in range(self.num_neurons):
            print("neuron " + str(i))
            self.neurons[i].print_neuron()
            print("")

class NeuralNetwork:
    def __init__(self, num_layers, num_neurons, activation, 
                       num_inputs, loss, eta, weights):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.loss = loss
        self.eta = eta

        self.activation = []
        for i in range(num_layers):
            self.activation.append(activation)

        self.layers = []
        prev_inputs = self.num_inputs
        if weights == "random":
            for i in range(self.num_layers):
                self.layers.append(FullyConnectedLayer(self.num_neurons[i],
                                                       self.activation[i],
                                                       prev_inputs,
                                                       self.eta,
                                                       "random"))
                prev_inputs = self.num_neurons[i]
        else:
            for i in range(self.num_layers):
                self.layers.append(FullyConnectedLayer(self.num_neurons[i],
                                                       self.activation[i],
                                                       prev_inputs,
                                                       self.eta,
                                                       weights[i]))
                prev_inputs = self.num_neurons[i]

    def calculate(self, data):
        output = data
        for layer in self.layers:
            output = layer.calculate(output)
            if debug: print(output)
        return output

    def calculateloss(self, prediction, target):
        return self.loss(prediction, target)

    def train(self, data, target):
        prediction = self.calculate(data)
        if debug: print("prediction: " + str(prediction))

        derivs = []
        for i in range(len(prediction)):
            derivs.append(get_deriv(self.loss)(prediction[i],target[i]))
        if debug: print("derivs: " + str(derivs))

        for i in range(self.num_layers-1,-1,-1):
            derivs = self.layers[i].train(derivs)
            if debug: print("derivs for layer " + str(i) + ": " + str(derivs))

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

    if sys.argv[1] == 'example':
        print("Running example from class.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [2,2]
        weights = [[[.15,.20,.35],[.25,.30,.35]],[[.40,.45,.60],[.50,.55,.60]]]

        nn = NeuralNetwork(num_layers, num_neurons, "logistic", 
                           num_inputs, square_error, .5, weights)

        print("Loss before training example: ", end = '')
        print(nn.calculateloss(nn.calculate([.05,.10]),[.01,.99]))

        nn.train([.05,.10],[.01,.99])

        print("Loss after training example: ", end = '')
        print(nn.calculateloss(nn.calculate([.05,.10]),[.01,.99]))

    elif sys.argv[1] == 'and':
        print("Running 'and' example.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [1,1]
        weights = "random"

        nn = NeuralNetwork(num_layers, num_neurons, "logistic", 
                           num_inputs, square_error, .1, weights)

        inputs = [([0, 0], [0]), ([0, 1], [0]), ([1, 0], [0]), ([1, 1], [1])]
        for i in range(250000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))
        
    else:
        print("Running 'xor' example from online.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [2,1]
        weights = [[[-0.06782947598673161,0.2214514234604232,-0.4654700884762584],[0.9487814395569221,0.4662836664076017,0.10219816991955463]],[[-0.21256111621528748,0.6039091636457407,0.8141837643885104]]]
        nn = NeuralNetwork(num_layers, num_neurons, logistic, 
                           num_inputs, square_error, .2, weights)
        #print(nn.calculate([0,1]))
        inputs = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        data = [(nn.calculate(x),y) for x,y in inputs]
        #nn.train([0,1],[1])
        print("mse before: " + str(mse(data)))
        for i in range(8000):
            for sample, target in inputs:
                nn.train(sample, target)
            data = [(nn.calculate(x),y) for x,y in inputs]
            print("mse after: " + str(mse(data)))

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))
        exit()
        nn.print_nn()
        exit()
        
        print("Running 'xor' example.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [1,1]
        weights = "random"

        print("Training with one perceptron.")
        inputs = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        nn = NeuralNetwork(num_layers, num_neurons, "logistic", 
                           num_inputs, square_error, .1, weights)

        for i in range(250000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))

        num_inputs = 2
        num_layers = 3
        num_neurons = [3,3,1]
        weights = "random"

        print("Training with multiple perceptrons.")
        inputs = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
        nn = NeuralNetwork(num_layers, num_neurons, "logistic", 
                           num_inputs, square_error, .1, weights)

        for i in range(250000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))

if __name__ == '__main__':
    main()


