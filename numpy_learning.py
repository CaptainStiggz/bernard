import numpy as np
from scipy import ndimage
import skimage

# for learning
def numpy_test():

   a = np.array([1, 1, 1, 2, 2, 2, 3, 4, 5])
   b = np.array([0, 11, 12, 13, 14, 15])
   #print(b[np.arange(a.shape[0][:, None]), a]) # throws error

   # [11 11 11 12 12 12 13 14 15]

   # a = np.array([1, 1, 1, 2, 2, 2, 3, 4, 5, 13, 14, 15])
   a = a.reshape(3, 3)
   print("in:")
   print(a)

   # in:
   # [[1 1 1]
   #  [2 2 2]
   #  [3 4 5]]

   b = skimage.util.shape.view_as_windows(a, (2, 2))
   print(b)

   # [[[[1 1]
   #    [2 2]]

   #   [[1 1]
   #    [2 2]]]


   #  [[[2 2]
   #    [3 4]]

   #   [[2 2]
   #    [4 5]]]]

   ar2D = b.reshape(-1, b.shape[-2]*b.shape[-1])
   print(ar2D)

   # rows are the sub-matrices (last 2 axes of b)
   # [[1 1 2 2]
   #  [1 1 2 2]
   #  [2 2 3 4]
   #  [2 2 4 5]]

   # bincount2d_vectorized
   c = bincount2D_vectorized(ar2D)
   #u = unique2D_vectorized(ar2D)
   p = c / 4
   p[p == 0] = 1
   print(p)
   l = np.log(p)
   s = np.sum(l, axis=1)
   print("pdf sum:")
   print(s)

   print(c)

   # [[0 2 2 0 0 0]
   #  [0 2 2 0 0 0]
   #  [0 0 2 1 1 0]
   #  [0 0 2 0 1 1]]

   # temp = c[c > 1] = 1
   # print(temp)

   print(c[np.arange(ar2D.shape[0])[:, None], ar2D])

   # [[2 2 2 2]
   #  [2 2 2 2]
   #  [2 2 1 1]
   #  [2 2 1 1]]

   out = c[np.arange(ar2D.shape[0])[:, None], ar2D].reshape(b.shape)
   print("out:")
   print(out)

   # out:
   # [[[[2 2]
   #    [2 2]]

   #   [[2 2]
   #    [2 2]]]


   #  [[[2 2]
   #    [1 1]]

   #   [[2 2]
   #    [1 1]]]]

def concatenate_test():
   a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
   b = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
   a = a.reshape(3, 3)
   b = b.reshape(3, 3)

   print(np.concatenate([a, b], axis=1))

def bincount_test():
   a = np.array([[1, 2, 2, 1], [3, 4, 5, 6]])
   print(a)
   b = bincount2D_vectorized(a)
   print(b)

# Vectorized solution (simple but non-performant)
def bincount2D_vectorized(a):
   print("bincount2D_vectorized:")
   N = a.max() + 1
   print(N)

   # 6 

   a_offs = a + np.arange(a.shape[0])[:, None] * N
   print(np.arange(a.shape[0])[:, None] * N)

   # [[ 0]
   #  [ 6]
   #  [12]
   #  [18]]

   print(a_offs)

   # [[ 1  1  2  2]
   #  [ 7  7  8  8]
   #  [14 14 15 16]
   #  [20 20 22 23]]

   print(a_offs.ravel())

   # [ 1  1  2  2  7  7  8  8 14 14 15 16 20 20 22 23]

   print(np.bincount(a_offs.ravel(), minlength=a.shape[0]*N))

   # [0 2 2 0 0 0 0 2 2 0 0 0 0 0 2 1 1 0 0 0 2 0 1 1]

   return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1, N)

# vectorized unique (0s elsewhere)
def unique2D_vectorized(a):
   print("unique2D_vectorized:")
   N = a.max() + 1
   a_offs = a + np.arange(a.shape[0])[:, None] * N
   print(np.unique(a_offs.ravel()))
   #return np.unique(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1, N)

# MAIN
# -----------

# numpy_test()
bincount_test()