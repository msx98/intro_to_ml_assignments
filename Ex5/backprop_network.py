"""

backprop_network.py

"""



import random

import numpy as np

from scipy.special import softmax

import math
irange = lambda a, b: range(a, b+1) if b >= a else range(a, b-1, -1)


class Network(object):



    def __init__(self, sizes):

        """The list ``sizes`` contains the number of neurons in the

        respective layers of the network.  For example, if the list

        was [2, 3, 1] then it would be a three-layer network, with the

        first layer containing 2 neurons, the second layer 3 neurons,

        and the third layer 1 neuron.  The biases and weights for the

        network are initialized randomly, using a Gaussian

        distribution with mean 0, and variance 1.  Note that the first

        layer is assumed to be an input layer, and by convention we

        won't set any biases for those neurons, since biases are only

        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)

                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.reset_variables()
    
    def reset_variables(self):
        self.skip_training_loss = False
        self.training_data = None
        self.test_data = None

        # code below added to avoid unnecessary reallocation
        self.v = [None for _ in range(self.num_layers)]
        self.z = [None for _ in range(self.num_layers)]
        self.d = [None for _ in range(self.num_layers)]
        self.db = [None for _ in range(self.num_layers-1)]
        self.dw = [None for _ in range(self.num_layers-1)]

        # keep training loss, accuracy, and test accuracy
        self.training_loss = []
        self.training_accuracy = []
        self.test_accuracy = []
    
    def add_loss(self):
        if not self.skip_training_loss:
            self.training_loss.append(self.loss(self.training_data))
        if not self.skip_training_loss:
            self.training_accuracy.append(self.one_hot_accuracy(self.training_data))
        if not self.skip_training_loss:
            self.test_accuracy.append(self.one_label_accuracy(self.test_data))


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,

            test_data):

        """Train the neural network using mini-batch stochastic

        gradient descent.  The ``training_data`` is a list of tuples

        ``(x, y)`` representing the training inputs and the desired

        outputs.  """

        self.training_data = training_data
        self.test_data = test_data

        self.add_loss()

        print("Initial test accuracy: {0}".format(self.test_accuracy[-1]))

        n = len(training_data)

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [

                training_data[k:k+mini_batch_size]

                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:

                self.update_mini_batch(mini_batch, learning_rate)

            self.add_loss()
            print ("Epoch {0} test accuracy: {1}".format(j, self.test_accuracy[-1]))







    def update_mini_batch(self, mini_batch, learning_rate):

        """Update the network's weights and biases by applying

        stochastic gradient descent using backpropagation to a single mini batch.

        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw

                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb

                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 

        vector x and a one-hot vector y.

        The function should return a tuple of two lists (db, dw) 

        as described in the assignment pdf. """
        
        # db[l] := derivative of loss with respect to biases in layer l+1
        # dw[l] := derivative of loss with respect to weights in layer l+1
        
        w, b, L = self.weights, self.biases, self.num_layers

        # forward pass
        v = self.v
        z = self.z
        v[0] = None
        z[0] = x
        for l in irange(0, L-2):
            v[l+1] = w[l+1   +(-1)]@z[l] + b[l+1  +(-1)]
            z[l+1] = relu(v[l+1])
        v[L-1] = w[L-1  +(-1)]@z[L-2] + b[L-1   +(-1)]
        z[L-1] = softmax(v[L-1])

        # backward pass
        d = self.d
        d[L-1] = z[L-1] - y
        d[L-2] = w[L-2].T @ d[L-1]
        for l in range(L-3, 0, -1): # l in range L-3, ..., 1
            d[l] = w[l].T @ (d[l+1] * relu_derivative(v[l]))
        
        # db[i] = dl/d(b[i+1])
        db = self.db
        dw = self.dw
        for l in range(0, L-1):
            db[l] = d[l+1] * relu_derivative(v[l+1])
            dw[l] = db[l] @ z[l].T
        
        return db, dw

    def one_label_accuracy(self, data):

        """Return accuracy of network on data with numeric labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), y)

         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))



    def one_hot_accuracy(self,data):

        """Return accuracy of network on data with one-hot labels"""

        output_results = [(np.argmax(self.network_output_before_softmax(x)), np.argmax(y))

                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))





    def network_output_before_softmax(self, x):

        """Return the output of the network before softmax if ``x`` is input."""

        layer = 0

        for b, w in zip(self.biases, self.weights):

            if layer == len(self.weights) - 1:

                x = np.dot(w, x) + b

            else:

                x = relu(np.dot(w, x)+b)

            layer += 1

        return x



    def loss(self, data):

        """Return the CE loss of the network on the data"""

        loss_list = []

        for (x, y) in data:

            net_output_before_softmax = self.network_output_before_softmax(x)

            net_output_after_softmax = self.output_softmax(net_output_before_softmax)

            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))



    def output_softmax(self, output_activations): # i.e. v[L] -> z[L]

        """Return output after softmax given output before softmax"""

        return softmax(output_activations) # i.e. z[L]



    def loss_derivative_wr_output_activations(self, output_activations, y): # i.e. v[L], y -> dl(b,y,w,z)/d(v[L])

        """Return the derivative of the loss wrt the output activations"""

        return self.output_softmax(output_activations) - y # i.e. z[L] - y





def relu(z: np.array):
    assert (z.ndim == 1) or (z.ndim == 2 and z.shape[1] == 1) or (z.ndim == 2 and z.shape[0] == 1)
    return np.maximum(z,0)



def relu_derivative(z):
    # ReLU := max(0, z)
    # z > 0 -> Relu = z -> Relu' = 1
    # z = 0 -> Relu = 0 -> Relu' = lim h->0 (max(0, z+h) - max(0, z)) / h = lim h->0 (h - 0) / h = 1
    # z < 0 -> Relu = 0 -> Relu' = 0
    assert (z.ndim == 1) or (z.ndim == 2 and z.shape[1] == 1) or (z.ndim == 2 and z.shape[0] == 1)
    return np.where(z >= 0, 1, 0)


class ListPtr:
    def __init__(self, arr_or_length, offset=0):
        if type(arr_or_length) == int:
            self.arr = [None for _ in range(arr_or_length)]
        else:
            self.arr = arr_or_length
        self.offset = offset
    def __getitem__(self, key):
        return self.arr[key + self.offset]