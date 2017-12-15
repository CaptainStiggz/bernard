"""
network.py
--------------------------

Base neural network class
Much of the starting code informed by: http://neuralnetworksanddeeplearning.com/chap6.html
"""

import numpy as np
import time
import gzip
import _pickle as cPickle
import random

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

# timeit decorator
def timeit(method):
   def timed(*args, **kwargs):
      ts = time.time()
      result = method(*args, **kwargs)
      te = time.time()

      print("%r %2.2f ms"%(method.__name__, (te - ts) * 1000))
      # if 'log_time' in kw:
      #    name = kw.get('log_name', method.__name__.upper())
      #    kw['log_time'][name] = int((te - ts) * 1000)
      # else:
      #    print '%r  %2.2f ms' % \
      #       (method.__name__, (te - ts) * 1000)
      return result
   return timed

#### Load the MNIST data
def load_data_shared(filename="../../data/mnist.pkl.gz"):
   f = gzip.open(filename, 'rb')
   training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
   f.close()

   return [training_data, validation_data, test_data]
   # def shared(data):
   #    """Place the data into shared variables.  This allows Theano to copy
   #    the data to the GPU, if one is available.

   #    """
   #    shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
   #    shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
   #    return shared_x, T.cast(shared_y, "int32")
   
   # return [shared(training_data), shared(validation_data), shared(test_data)]

def load_mnist_data():
   def vectorized_result(j):
      """Return a 10-dimensional unit vector with a 1.0 in the jth
      position and zeroes elsewhere.  This is used to convert a digit
      (0...9) into a corresponding desired output from the neural
      network."""
      e = np.zeros((10, 1))
      e[j] = 1.0
      return e

   tr_d, va_d, te_d = load_data_shared()
   training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
   training_results = [vectorized_result(y) for y in tr_d[1]]
   training_data = list(zip(training_inputs, training_results))
   validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
   validation_data = list(zip(validation_inputs, va_d[1]))
   test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
   test_data = list(zip(test_inputs, te_d[1]))
   return (training_data, validation_data, test_data)

def sigmoid_prime(z):
   """Derivative of the sigmoid function."""
   return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
   """The sigmoid function"""
   return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
   """Derivative of the sigmoid function"""
   return sigmoid(z) * (1 - sigmoid(z))

class FullyConnectedLayer(object):

   def __init__(self, n_io, activation_fn=sigmoid, p_dropout=0.0):
      self.n_in = n_io[0]
      self.n_out = n_io[1]
      self.activation_fn = activation_fn
      self.p_dropout = p_dropout

      # TODO: sllow different weigh initialization techniques
      self.w = np.random.normal(loc=0.0, scale=np.sqrt(1.0/self.n_out), size=(self.n_out, self.n_in))
      self.b = np.random.normal(loc=0.0, scale=1.0, size=(self.n_out, 1))

class Network(object):

   def __init__(self, sizes, layers):
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
      self.num_layers = len(layers) + 1 # len(sizes) #len(layers)
      self.layers = layers
      self.biases = []
      self.weights = []
      
      for layer in layers:
         self.biases.append(layer.b)
         self.weights.append(layer.w)

      #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
      #self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

   def feedforward(self, a):
      """Return the output of the network if ``a`` is input."""
      for b, w in zip(self.biases, self.weights):
         a = sigmoid(np.dot(w, a)+b)
      return a

   def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
      """Train the neural network using mini-batch stochastic
      gradient descent.  The ``training_data`` is a list of tuples
      ``(x, y)`` representing the training inputs and the desired
      outputs.  The other non-optional parameters are
      self-explanatory.  If ``test_data`` is provided then the
      network will be evaluated against the test data after each
      epoch, and partial progress printed out.  This is useful for
      tracking progress, but slows things down substantially."""
      if test_data: n_test = len(test_data)
      n = len(training_data)
      for j in range(epochs):
         random.shuffle(training_data)
         mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
         for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)
         if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
         else:
            print("Epoch {0} complete".format(j))

   def update_mini_batch(self, mini_batch, eta):
      """Update the network's weights and biases by applying
      gradient descent using backpropagation to a single mini batch.
      The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
      is the learning rate."""
      nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
      nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

      for x, y in mini_batch:
         delta_nabla_b, delta_nabla_w = self.backprop(x, y)
         nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
         nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

      for i in range(len(self.layers)):
         self.layers[i].w -= (eta / len(mini_batch)) * nabla_w[i]
         self.layers[i].b -= (eta / len(mini_batch)) * nabla_b[i]

      self.weights = [layer.w for layer in self.layers]
      self.biases = [layer.b for layer in self.layers]
      #self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
      #self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

   def backprop(self, x, y):
      """Return a tuple ``(nabla_b, nabla_w)`` representing the
      gradient for the cost function C_x.  ``nabla_b`` and
      ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
      to ``self.biases`` and ``self.weights``."""
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      # feedforward
      activation = x
      activations = [x] # list to store all the activations, layer by layer
      zs = [] # list to store all the z vectors, layer by layer
      for b, w in zip(self.biases, self.weights):
         z = np.dot(w, activation)+b
         zs.append(z)
         activation = sigmoid(z)
         activations.append(activation)
      # backward pass
      delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
      nabla_b[-1] = delta
      nabla_w[-1] = np.dot(delta, activations[-2].transpose())
      # Note that the variable l in the loop below is used a little
      # differently to the notation in Chapter 2 of the book.  Here,
      # l = 1 means the last layer of neurons, l = 2 is the
      # second-last layer, and so on.  It's a renumbering of the
      # scheme in the book, used here to take advantage of the fact
      # that Python can use negative indices in lists.
      for l in range(2, self.num_layers):
         z = zs[-l]
         sp = sigmoid_prime(z)
         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
         nabla_b[-l] = delta
         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
      return (nabla_b, nabla_w)

   def evaluate(self, test_data):
      """Return the number of test inputs for which the neural
      network outputs the correct result. Note that the neural
      network's output is assumed to be the index of whichever
      neuron in the final layer has the highest activation."""
      test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
      return sum(int(x == y) for (x, y) in test_results)

   def cost_derivative(self, output_activations, y):
      """Return the vector of partial derivatives \partial C_x /
      \partial a for the output activations."""
      return (output_activations-y)

# class Network(object):
#    """The overarching network class"""

#    def __init__(self, sizes, layers, mini_batch_size=1):
#       """Load the layers defining the network architecture"""

#       self.layers = layers
#       self.mini_batch_size = mini_batch_size

#    def feedforward(self, a):
#       for layer in self.layers:
#          a = layer.activation_fn(np.dot(layer.w, a) + layer.b)
#          # a = layer.activation_fn(np.dot(a, layer.w) + layer.b)
#          # equivalently a = layer.activation_fn(np.dot(np.transpose(layer.w), a) + layer.b)

#       # print("output shape:")
#       # print(a.shape)
#       return a

#    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
#       """Train the neural network using mini-batch stochastic
#       gradient descent.  The ``training_data`` is a list of tuples
#       ``(x, y)`` representing the training inputs and the desired
#       outputs.  The other non-optional parameters are
#       self-explanatory.  If ``test_data`` is provided then the
#       network will be evaluated against the test data after each
#       epoch, and partial progress printed out.  This is useful for
#       tracking progress, but slows things down substantially."""

#       if test_data: n_test = len(test_data)
#       n = len(training_data)

#       for j in range(epochs):
#          random.shuffle(training_data)
#          mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
#          for mini_batch in mini_batches:
#             self.update_mini_batch(mini_batch, eta)
#          if test_data:
#             print("Epoch {0}, {1}/{2}".format(j, self.evaluate(test_data), n_test))
#          else:
#             print("Epoch {0} complete".format(j))

#    def update_mini_batch(self, mini_batch, eta):
#       """Update the network's weights and biases by applying
#       gradient descent using backpropagation to a single mini batch.
#       The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
#       is the learning rate."""

#       nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
#       nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

#       for x, y in mini_batch:
#          delta_nabla_b, delta_nabla_w = self.backprop(x, y)
#          nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
#          nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

#       for i in range(len(self.layers)):
#          layer = self.layers[i]
#          layer.w = layer.w - (eta / len(mini_batch)) * nabla_w[i] #for w,nw in zip(layer.w, nabla_w)]
#          layer.b = layer.b - (eta / len(mini_batch)) * nabla_b[i] #for b,nb in zip(layer.b, nabla_b)]

#    def backprop(self, x, y):
#       """Return a tuple ``(nabla_b, nabla_w)`` representing the
#       gradient for the cost function C_x.  ``nabla_b`` and
#       ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
#       to ``self.biases`` and ``self.weights``."""

#       nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
#       nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

#       # feedforward
#       activation = x #np.zeros(784,)
#       # print("activation shape:")
#       # print(activation.shape)
#       activations = [x] # list to store all the activations, layer by layer
#       zs = [] # list to store all the z vectors, layer by layer

#       for layer in self.layers:
#          # print("layer.w.shape:")
#          # print(layer.w.shape)
#          # print("layer.b.shape:")
#          # print(layer.b.shape)
#          z = np.dot(layer.w, activation) + layer.b
#          # print("z.shape:")
#          # print(z.shape)
#          zs.append(z)
#          activation = layer.activation_fn(z)
#          activations.append(activation)

#       # backward pass
#       delta = self.d_cost(activations[-1], y) * d_sigmoid(zs[-1])
#       nabla_b[-1] = delta
#       nabla_w[-1] = np.dot(delta, activations[-2].transpose())

#       for l in range(2, len(self.layers)+1):
#          z = zs[-l]
#          sp = d_sigmoid(z)
#          delta = np.dot(self.layers[-l+1].w.transpose(), delta) * sp
#          nabla_b[-l] = delta
#          nabla_w[-l] = np.dot(delta, activations[-l+1].transpose())

#       # print("nabla_b.shape:")
#       # print(len(nabla_b))
#       # print("nabla_w.shape:")
#       # print(len(nabla_w))
#       return (nabla_b, nabla_w)

#    def evaluate(self, test_data):
#       """Return the vector of partial derivatives \partial C_x /
#       \partial a for the output activations."""

#       test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
#       return sum(int(x == y) for (x, y) in test_results)

#    def d_cost(self, output_activations, y):
#       return (output_activations - y)

@timeit
def test():
   training_data, validation_data, test_data = load_mnist_data()

   net = Network([784, 30, 10], [FullyConnectedLayer((784, 30)), FullyConnectedLayer((30, 10))])
   #net = Network([784, 30, 10])
   net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
   #print(net.feedforward(np.random.normal(loc=0.0, scale=1.0, size=(784,1))))

test()

# net = network.Network([FullyConnectedLayer(784, 30), FullyConnectedLayer(30, 10)])
# print(network.feedforward(np.random))