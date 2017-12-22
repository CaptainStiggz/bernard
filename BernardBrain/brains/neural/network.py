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
import math

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

global_dtype = np.float64

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

def sigmoid(z):
   """The sigmoid function"""
   return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(z):
   """Derivative of the sigmoid function"""
   return sigmoid(z) * (1.0 - sigmoid(z))

# def ReLU(z):
#    """The rectified linear unit (ReLU) function"""
#    return np.maximum(z, 0.0)
#    #z[z < 0] = 0
#    #return z

# def d_ReLU(z):
#    """Derivative of the ReLU function"""
#    z[z < 0.0] = 0.0
#    z[z > 0.0] = 1.0
#    return z
#    #return 0 if (z <= 0) else 1

# def d_quadratic(a, y):
#    """Derivative of the quadratic cost function"""
#    t = (a - y)
#    print("ACTIVATION:")
#    print(a)
#    print("Y:")
#    print(y)
#    print("COST:")
#    print(t)
#    return t

# def d_cross_entropy(a, y):
#    """Derivative of the cross entropy cost function"""
#    #return (a - y) / (a * a - a)
#    # print(np.divide((a - y), (np.square(a) - a)))
#    t = np.divide((a - y), (np.square(a) - a))
#    print("ACTIVATION:")
#    print(a)
#    print("Y:")
#    print(y)
#    print("COST:")
#    print(t)
#    return t
#    #return (a - y)
#    #return np.divide((a - y), (np.square(a) - a))
#    # print(a.shape)
#    #return np.true_divide((a - y), (np.multiply(a, a) - a))

# def d_activation(activation_fn):
#    if activation_fn.__name__ == "sigmoid":
#       return d_sigmoid
#    elif activation_fn.__name__ == "ReLU":
#       return d_ReLU
#    else:
#       raise Exception("Unrecognized activation function.")

# def d_cost(cost_fn):
#    if cost_fn.__name__ == "quadratic":
#       return d_quadratic
#    elif cost_fn.__name__ == "cross_entropy":
#       return d_cross_entropy
#    else:
#       raise Exception("Unrecognized cost function.")

#### Define the activation functions

class SigmoidActivation(object):

   @staticmethod
   def fn(z):
      """Return the sigmoid activation function"""
      return sigmoid(z)

   @staticmethod
   def df(z):
      """Return the derivative of the sigmoid activation function"""
      return sigmoid(z) * (1.0 - sigmoid(z))

class ReLUActivation(object):

   @staticmethod
   def fn(z):
      """Return the ReLU activation function"""
      z[z <= 0.0] = 0.0
      return z

   @staticmethod
   def df(z):
      """Return the derivative of the ReLU activation function"""
      z[z <= 0.0] = 0.0
      z[z > 0.0] = 1.0
      return z

class SoftmaxActivation(object):

   @staticmethod
   def fn(z):
      """Return the softmax function. Note that we shift it for numerical 
      stability to avoid overflowing float64 datatype."""
      
      # TODO: this is kind of a hacky if-check

      if len(z.shape) < 3: # serial version
         expz = np.exp(z - np.max(z))
         return expz / np.sum(expz)

      else: # parallel version
         expz = np.exp((z.transpose(1, 0, 2) - np.max(z, axis=1)).transpose(1, 0, 2))
         sumz = np.sum(expz, axis=1)
         return (expz.transpose(1, 0, 2) / sumz).transpose(1, 0, 2)

   @staticmethod
   def df(z):
      """Return the derivative of the softmax function"""

      #return 1.0

      # TODO: if we want to use softmax as an intermediate layer
      if len(z.shape) < 3: # serial version
         expz = np.exp(z - np.max(z))
         sumz = np.sum(expz)

         # calculate Jacobian da/dz
         inv_identity = np.ones((z.shape[0], z.shape[0])) - np.identity(z.shape[0])
         J1 = -(expz.transpose() * expz / sumz) * inv_identity # off-diagonal
         J2 = (((expz * sumz) * (sumz - expz)) / (sumz * sumz * sumz)) * np.identity(z.shape[0]) # diagonal
         J = J1 + J2
         d = J.dot(z)
         return d

      else: # parallel version
         pass
      
      return 1

#### Define the cost functions

class QuadraticCost(object):

   @staticmethod
   def fn(a, y):
      """Return the quadratic cost associated with an output ``a`` and 
      desired output ``y``."""
      return 0.5 * np.linalg.norm(a - y)**2

   @staticmethod
   def delta(z, a, y):
      """Return the error delta from the output layer."""
      return (a - y) * d_sigmoid(z)


class CrossEntropyCost(object):

   @staticmethod
   def fn(a, y):
      """Return the cost associated with an output ``a`` and desired output
      ``y``.  Note that np.nan_to_num is used to ensure numerical
      stability.  In particular, if both ``a`` and ``y`` have a 1.0
      in the same slot, then the expression (1-y)*np.log(1-a)
      returns nan.  The np.nan_to_num ensures that that is converted
      to the correct value (0.0)."""
      return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

   @staticmethod
   def delta(z, a, y):
      """Return the error delta from the output layer.  Note that the
      parameter ``z`` is not used by the method.  It is included in
      the method's parameters in order to make the interface
      consistent with the delta method for other cost classes."""
      return (a - y)

class LogLikelihoodCost(object):

   @staticmethod
   def fn(a, y):
      """Return the cost associated with an output ``a``, where a is
      the probability distribution corresponding to the correct output
      ``y``. For example, a[0] is the probability the output corresponds
      to y[0]"""
      return np.sum(np.nan_to_num(-np.log(a)))

   @staticmethod
   def delta(z, a, y):
      """Return the error delta from the output layer."""
      return (a - y)

#### Define the cost functions

class NoRegularization(object):

   @staticmethod
   def term(w, eta, n):
      """Return the weight regularization term (unmodified here)
      where ``w`` is the weights to be updated, ``eta`` is the learning 
      rate, and ``n`` is the length of the training data. """
      return w

class L2Regularization(object):

   def __init__(self, lmbda):
      self.lmbda = lmbda

   def term(self, w, eta, n):
      """Return the weight regularization term via L2 regularization
      where ``w`` is the weights to be updated, ``eta`` is the learning 
      rate, and ``n`` is the length of the training data. """
      return (1 - eta * (self.lmbda / n)) * w

#### Define the available layer types

class FullyConnectedLayer(object):

   def __init__(self, n_in, n_out, activation=SigmoidActivation, cost=CrossEntropyCost, p_dropout=0.0):
      self.n_in = n_in
      self.n_out = n_out
      self.activation = activation
      self.cost = cost
      self.p_dropout = p_dropout

      # TODO: sllow different weight initialization techniques
      self.w = np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_out, n_in)).astype(global_dtype)
      self.b = np.random.normal(loc=0.0, scale=1.0, size=(n_out, 1)).astype(global_dtype)

class SoftmaxLayer(object):

   def __init__(self, n_in, n_out, activation=SoftmaxActivation, cost=LogLikelihoodCost, p_dropout=0.0):
      self.n_in = n_in
      self.n_out = n_out
      self.activation = activation
      self.cost = cost
      self.p_dropout = p_dropout

      self.w = np.zeros((n_out, n_in), dtype=global_dtype)
      self.b = np.zeros((n_out, 1), dtype=global_dtype)

class Network(object):

   def __init__(self, layers, regularization=NoRegularization):
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
      self.regularization = regularization

   def feedforward(self, a):
      """Return the output of the network if ``a`` is input."""
      for layer in self.layers:
         a = layer.activation.fn(np.dot(layer.w, a) + layer.b)
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

      @timeit
      def train_epoch(training_data, mini_batch_size, eta, test_data):
         random.shuffle(training_data)
         mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
         for mini_batch in mini_batches:
            self.update_mini_batch_parallel(mini_batch, eta, len(training_data))
            #self.update_mini_batch_serial(mini_batch, eta)
         if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
         else:
            print("Epoch {0} complete".format(j))

      for j in range(epochs):
         train_epoch(training_data, mini_batch_size, eta, test_data)

   def update_mini_batch_parallel(self, mini_batch, eta, n):
      """Update the network's weights and biases by applying
      gradient descent using backpropagation to a single mini batch.
      The ``mini_batch`` is a list of tuples ``(x, y)``, ``eta``
      is the learning rate, and ``n`` is the length of the training data."""

      dnb, dnw = self.backprop_parallel(mini_batch)
      for i in range(len(self.layers)):
         rw = self.regularization.term(self.layers[i].w, eta, n)
         self.layers[i].w = rw - (eta / len(mini_batch)) * np.sum(dnw[i], axis=0)
         self.layers[i].b -= (eta / len(mini_batch)) * np.sum(dnb[i], axis=0)

   def backprop_parallel(self, mini_batch):
      """Return a tuple ``(nabla_b, nabla_w)`` representing the
      gradient for the cost function C_x.  ``nabla_b`` and
      ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
      to ``self.biases`` and ``self.weights``."""

      x = np.array([x for x, y in mini_batch], dtype=global_dtype)
      y = np.array([y for x, y in mini_batch], dtype=global_dtype)

      nabla_b = [np.zeros((len(mini_batch),)+layer.b.shape, dtype=global_dtype) for layer in self.layers]
      nabla_w = [np.zeros((len(mini_batch),)+layer.w.shape, dtype=global_dtype) for layer in self.layers]

      # TODO: any way to speed this up further?
      # feedforward
      activation = x
      activations = [x] # list to store all the activations, layer by layer
      zs = [] # list to store all the z vectors, layer by layer
      for layer in self.layers:
         # z = np.dot(layer.w, activation).transpose(1, 0, 2) + layer.b
         z = np.matmul(layer.w, activation) + layer.b
         zs.append(z)
         activation = layer.activation.fn(z)
         activations.append(activation)

      # TODO: any way to speed this up further?
      # backward pass
      L = self.layers[-1]
      z = zs[-1]
      a = activations[-1]
      delta = L.cost.delta(z, a, y)
      nabla_b[-1] = delta
      nabla_w[-1] = np.matmul(delta, activations[-2].transpose(0, 2, 1))
      for l in range(2, self.num_layers):
         w = self.layers[-l+1].w
         z = zs[-l]
         da = self.layers[-l].activation.df(z)
         delta = np.matmul(w.transpose(), delta) * da
         nabla_b[-l] = delta
         nabla_w[-l] = np.matmul(delta, activations[-l-1].transpose(0, 2, 1))

      return (nabla_b, nabla_w)      

   def update_mini_batch_serial(self, mini_batch, eta):
      """Update the network's weights and biases by applying
      gradient descent using backpropagation to a single mini batch.
      The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
      is the learning rate."""

      nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
      nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

      for x, y in mini_batch:
         delta_nabla_b, delta_nabla_w = self.backprop_serial(x, y)
         nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
         nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

      for i in range(len(self.layers)):
         self.layers[i].w -= (eta / len(mini_batch)) * nabla_w[i]
         self.layers[i].b -= (eta / len(mini_batch)) * nabla_b[i]

   def backprop_serial(self, x, y):
      """Return a tuple ``(nabla_b, nabla_w)`` representing the
      gradient for the cost function C_x.  ``nabla_b`` and
      ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
      to ``self.biases`` and ``self.weights``."""
      nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
      nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

      # feedforward
      activation = x
      activations = [x] # list to store all the activations, layer by layer
      zs = [] # list to store all the z vectors, layer by layer
      for layer in self.layers:
         z = np.dot(layer.w, activation) + layer.b
         zs.append(z)
         activation = layer.activation.fn(z)
         activations.append(activation)

      # backward pass
      L = self.layers[-1]
      z = zs[-1]
      a = activations[-1]
      #print(a.shape)
      delta = L.cost.delta(z, a, y)
      nabla_b[-1] = delta
      nabla_w[-1] = np.dot(delta, activations[-2].transpose())
      # Note that the variable l in the loop below is used a little
      # differently to the notation in Chapter 2 of the book.  Here,
      # l = 1 means the last layer of neurons, l = 2 is the
      # second-last layer, and so on.  It's a renumbering of the
      # scheme in the book, used here to take advantage of the fact
      # that Python can use negative indices in lists.
      for l in range(2, self.num_layers):
         w = self.layers[-l+1].w
         z = zs[-l]
         da = self.layers[-l].activation.df(z)
         delta = np.dot(w.transpose(), delta) * da
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

@timeit
def test():
   # a = np.arange(6.).reshape(2, 3)
   # b = np.arange(6.).reshape(3, 2)
   # print(np.dot(a, b))
   # c = np.array([b for x in range(10)])
   # print(a.shape)
   # print(c.shape)
   # print(np.dot(a, c).transpose(1, 0, 2))
   # print(np.matmul(a, c))

   # d = np.array([a for x in range(10)])
   # print(d.shape)
   # print(c.shape)
   # #print(np.tensordot(d, c, axes=[[1, 2], [1, 2]]).shape)
   # print(np.dot(d, c).shape)
   # print(np.matmul(d, c))
   # #print(np.tensordot(d, c).shape)
   # #print(np.dot(d, c).reshape(10, 2, 2))

   # a = np.arange(10.)
   # b = np.arange(10.)
   # print(a / b)
   #print(np.sum(a, axis=1))
   #b = np.arange(6.).reshape(2, 3)

   # z = np.array([[[1], [2], [3], [4], [5]], [[6], [7], [8], [9], [10]], [[11], [12], [13], [14], [15]]] )
   # # print(z.shape)
   # # print(len(z.shape)-2)
   # print(z)
   # print(np.sum(z, axis=1))
   # #print(z / np.sum(z, axis=1))
   # print((z.transpose(1, 0, 2) / np.sum(z, axis=1)).transpose(1, 0, 2))
   # # print(z.transpose(1, 0, 2))
   # # print(np.max(z, axis=1))
   # # print((z.transpose(1, 0, 2) - np.max(z, axis=1)).transpose(1, 0, 2))
   # # print(np.sum(z, axis=len(z.shape)-2))

   # z = np.array([[1], [2], [3], [4], [5]])
   # print(z.shape)
   # print(z.transpose().shape)
   # print(z.transpose() * z)
   # print(np.ones((3, 3)) - np.identity(3))

   training_data, validation_data, test_data = load_mnist_data()

   net = Network([
      
      # Hidden Layers
      FullyConnectedLayer(784, 30, activation=ReLUActivation),
      # FullyConnectedLayer(784, 30, activation_fn=ReLU),
      
      # Output layer
      #FullyConnectedLayer(30, 10, activation=SigmoidActivation, cost=CrossEntropyCost)
      SoftmaxLayer(30, 10)
   ], regularization=L2Regularization(5.0))

   # eta = 3.0 seems like a good value for SigmoidActivation
   # eta = 0.30 seems like an equivalent value for ReLUActivation
   net.SGD(training_data, 30, 10, 0.02, test_data=test_data)
   #print(net.feedforward(np.random.normal(loc=0.0, scale=1.0, size=(784,1))))

test()
