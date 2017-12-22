"""
bayes.py
--------------------------

Naive Bayes classification
Much of the starting code informed by: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

Other resources:
https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
http://dataaspirant.com/2017/02/06/naive-bayes-classifier-machine-learning/
http://blog.aylien.com/naive-bayes-for-dummies-a-simple-explanation/

A description of Pima institute parameters:
https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names

The scikit-learn informed versions from: http://kenzotakahashi.github.io/naive-bayes-from-scratch-in-python.html

"""

import csv
import random
import math
import numpy as np

def load_pima_data(filename='../../data/pima-indians-diabetes.data.csv', split_ratio=0.67):
   lines = csv.reader(open(filename, "rt", encoding="utf8"))
   dataset = list(lines)
   formatted_dataset = []
   for i in range(len(dataset)):
      x = [float(x) for x in dataset[i]]
      y = x[-1]
      del x[-1]
      formatted_dataset.append((np.array(x), y))
   
   random.shuffle(formatted_dataset)
   split_index = int(split_ratio * len(formatted_dataset))
   training_data = formatted_dataset[:split_index]
   test_data = formatted_dataset[split_index:]
   return (training_data, test_data)

# useful for text classification
class MultinomialNB(object):
   def __init__(self, alpha=1.0):
      self.alpha = alpha # laplace smoothing

   def fit(self, X, y):
      count_sample = X.shape[0]
      # TODO: this is clean, but slow for large ``y``
      separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
      # prior probabilities of each class. we use log(p(c)) to avoid floating point
      # underflow
      self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
      count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
      # log of probability of each feature log(p(t|c))
      self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
      return self

   def predict_log_proba(self, X):
      return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_ for x in X]

   def predict(self, X):
      return np.argmax(self.predict_log_proba(X), axis=1)

# similar to multinomial, but only takes binary values
# in the example of document classification, bernoulli just cares whether or not
# a word appears in the document. It doesn't care about frequency.
class BernoulliNB(object):
   def __init__(self, alpha=1.0, binarize=0.0):
      self.alpha = alpha
      self.binarize = binarize

   def fit(self, X, y):
      X = self._binarize_X(X)
      count_sample = X.shape[0]
      # group by class
      separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
      # class prior
      self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
      # count of each word
      count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha

      smoothing = 2 * self.alpha
      # number of documents in each class + smoothing
      n_doc = np.array([len(i) + smoothing for i in separated])
      self.feature_prob_ = count / n_doc[np.newaxis].T
      return self

   def predict_log_proba(self, X):
      X = self._binarize_X(X)
      return [(np.log(self.feature_prob_) * x + \
               np.log(1 - self.feature_prob_) * np.abs(x - 1)
               ).sum(axis=1) + self.class_log_prior_ for x in X]

   def predict(self, X):
      X = self._binarize_X(X)
      return np.argmax(self.predict_log_proba(X), axis=1)

   def _binarize_X(self, X):
      return np.where(X > self.binarize, 1, 0) if self.binarize != None else X

# for continuous variables
class GaussianNB(object):
   def __init__(self):
      pass

   def fit(self, X, y):
      separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
      self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in separated])
      return self

   def _prob(self, x, mean, std):
      exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
      return np.log(exponent / (np.sqrt(2 * np.pi) * std))

   def predict_log_proba(self, X):
      return [[sum(self._prob(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]

   def predict(self, X):
      return np.argmax(self.predict_log_proba(X), axis=1)

   def score(self, X, y):
      return sum(self.predict(X) == y) / len(y)

# First attempt, from scratch
class NaiveBayesClassifier(object):

   def __init__(self):
      pass

   def train(self, training_data):
      s = self.sort_dataset_by_class(training_data)
      return self.statistics_by_class(s)

   def run(self, training_data, test_data):
      s = self.train(training_data)
      predictions = self.predict_all_serial(test_data, s)
      accuracy = self.evaluate(predictions, test_data)
      p_correct = (accuracy / float(len(test_data)) * 100.0)
      print("Accuracy: {0}/{1} ({2}%)".format(accuracy, len(test_data), p_correct))

   def sort_dataset_by_class(self, training_data):
      """Returns a map relating output classes to numpy arrays. Each numpy array
      is m x n, where m is the number of inputs corresponding to that class, and n
      is the number of datapoints for each inpupt. ``training_data`` is a tuple of
      the form (input_vector, output_class)"""
      s = {}
      for x, y in training_data:
         if y not in s: s[y] = []
         s[y].append(x)

      for y, x in s.items():
         s[y] = np.array(x)
      return s

   def statistics_by_class(self, sorted_data):
      """Returns a map relating output classes to a 2-tuple, where the first tuple
      value is a vector corresponding to the mean of each feature in the input, 
      and the second value is the standard deviation of each feature in the input.
      ``sorted_data`` is a map of ouput classes to 2d numpy arrays, containing a
      'list' of vectors corresponding to the output class."""
      stats = {}
      for y, x in sorted_data.items():
         mean = np.mean(x, axis=0)
         stddev = np.std(x, axis=0)
         stats[y] = (mean, stddev)
      return stats


   # TODO: implement other models (multinomial, bernoulli)
   # TODO: implement log probabilities to avoid float underflow

   # P(H|Multiple Evidences) =  P(E1|H) * P(E2|H) * ... * P(En|H) * P(H) / P(Multiple Evidences)
   # for one input vector of evidence, P(Multiple Evidences) is the same, and can be ignored

   def gaussian_probability(self, x, mean, stddev):
      """Calculates the probability that ``x``, a vector of input features,
      corresponds to some output feature class, with mean and standard deviation
      vectors corresponding to the output class. Assumes a gaussian distribution."""
      expp = np.exp(-np.square(x - mean) / (2.0 * np.square(stddev)))
      dist = (1.0 / (np.sqrt(2.0 * np.pi) * stddev)) * expp
      return np.prod(dist)

   def calc_class_probabilities(self, x, class_statistics):
      """Calculates a map relating ouput classes to the probability that an
      input vector, ``x``, belongs to that class. Uses ``stats``, a map relating
      ouput classes to the means and standard deviations of input features."""
      probs = {}
      for y, stats in class_statistics.items():
         probs[y] = self.gaussian_probability(x, stats[0], stats[1])
      return probs

   def predict(self, x, class_statistics):
      """Returns the class label associated with the highest the probability that an
      input vector, ``x``, belongs to that class. Uses ``stats``, a map relating
      ouput classes to the means and standard deviations of input features."""
      probs = self.calc_class_probabilities(x, class_statistics)
      y_max, p_max = None, -1
      for y, p in probs.items():
         if y_max is None or p > p_max:
            y_max = y
            p_max = p
      return y_max

   def predict_all_serial(self, test_data, class_statistics):
      """Returns a list of predictions for each input in ``test_data``, a list
      of 2-tuples containing input vectors and output labels. Uses ``class_statisitcs``, 
      a map relating ouput classes to the means and standard deviations of input 
      features."""
      predictions = []
      for i in range(len(test_data)):
         predictions.append(self.predict(test_data[i][0], class_statistics))
      return predictions

   def evaluate(self, predictions, test_data):
      correct = 0
      for i in range(len(test_data)):
         if test_data[i][1] == predictions[i]:
            correct += 1
      return correct

def test():
   # training_data, test_data = load_pima_data()
   # classifier = NaiveBayesClassifier()
   # classifier.run(training_data, test_data)

   X = np.array([
      [2,1,0,0,0,0],
      [2,0,1,0,0,0],
      [1,0,0,1,0,0],
      [1,0,0,0,1,1]
   ])
   y = np.array([0,0,0,1])
   nb = MultinomialNB().fit(X, y)

test()