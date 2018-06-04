########################################################################
# This is 72 characters
########################################################################

import numpy as np
from scipy import ndimage
from numba import njit
from numba import prange
from skimage.util.shape import view_as_windows
import cv2
import time
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def timeit(method):
   """A decorator for timing functions. Outputs `function_name` and 
   corresponding elapsed time for decorated function calls.

   """
   
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

########################################################################
# Entropy Functions
########################################################################

@timeit
def image_diff(image1, image2):
   """Calculate the absolute pixel difference between two grayscale 
   images.
   
   Parameters
   ---------- 
   image1 : np.array (image)
   image2 : np.array (image)

   Returns
   ----------
   np.array : uint8
      A numpy array representing the absolute pixel difference 
      between two images. The output image is "normalized" to values
      between 0 and 1, where 0 represents no difference, and 1
      represents the largest individual pixel difference.

   """

   # Casting to floats produces a less noisy output than calculating
   # the difference of ints.
   diff = np.absolute(image2.astype(dtype=np.float) 
                      - image1.astype(dtype=np.float))
   return diff.astype(dtype=np.uint8)

@timeit
def entropy_local(image, ksize=9, step=1):
   """Calculate the local entropy of an image. Local entropy is the 
   entropy of a windowed region, relative to that window.
   
   Notes
   ----------
   `ksize` = 30 seems good for course detail (320x320ish pixels)
   `kSize` = 9 seems good for finer detail

   Parameters
   ---------- 
   image : np.array (image)
   ksize: int
      The size of the window kernel. This should be and odd
      integer greater than 1.
   step: int
      The window step size. This should be a an integer greater than 0.

   Returns
   ----------
   np.array : uint8
      A numpy array representing the entropy-filtered image. Each pixel 
      represents the entropy of the corresponding pixel in the original
      image, where the entropy value is calculated in a 
      (`ksize` x `ksize`) region. Values are normalized such that white 
      represents the regions of highest entropy, and black represents 
      regions of lowest entropy. Calculations are relative to the image.

   """

   if (not ksize % 2) or ksize < 3:
      raise Exception("ksize an odd integer greater than 1")
   if (step < 1):
      raise Exception("step must be an integer greater than 0")

   # Pad the image with zeroes.
   pad = math.floor(ksize / 2)
   a = np.pad(image, (pad, pad), 'constant', constant_values=(0, 0))

   # Get the windowed array of shape (a.height, a.width, ksize, ksize).
   b = view_as_windows(a, (ksize, ksize), step)
   
   # Replace each element of ksize x ksize kernel with the number of 
   # occurances of that element in the kernel.
   ar2D = b.reshape(-1, b.shape[-2] * b.shape[-1])
   c = bincount2D_vectorized(ar2D)
   #c = bincount2D_numba(ar2D, use_parallel=True, use_prange=True)
   
   # Convert bincount to probabilities and multiply by log2(p)
   p = c / (ksize * ksize)
   p[p == 0] = 1
   s = p * np.log2(p)

   # Replace each window with the sum of values in the window, and
   # reshape to the size of the original image (if step size = 1) or
   # scale down if step size > 1.
   new_shape = tuple(math.ceil(ti/step) for ti in image.shape)
   e = -np.sum(s, axis=1).reshape(new_shape)
   # This is an interesting measure...
   # e = -np.sum(p, axis=1).reshape(image.shape) 

   return normalized_image(e)

########################################################################
@timeit
def entropy_stack(image):
   """Generates a list of entropy-filtered images. using `entropy_local`,
   of various kernel and step sizes.

   Parameters
   ---------- 
   image : np.array (image)

   Returns
   ----------
   list : np.array : uint8
      A standard python list of images. Images are filtered by entropy,
      and contain low, medium, and high resolution results.

   """

   hi_rez = entropy_local(image, 7)
   medh_rez = entropy_local(image, 15, 8)
   medl_rez = entropy_local(image, 31, 16)
   low_rez = entropy_local(image, 63, 40)

   return [hi_rez, medh_rez, medl_rez, low_rez]

# kSize = 30 seems good for course detail (320x320ish pixels)
# kSize = 9 seems good for finer detail
# outputs a black/white entropy image
@timeit
def entropy_global(image, ksize = 9):
   if (not ksize % 2) or ksize < 3:
      raise Exception("Kernel must be of odd size > 1")

   # calculate probability histogram
   pdf = probability_distribution(image)

   # calculate intermediary entropy values
   partial = shannon_units(image, pdf)

   # calculate mean with convolve
   # seperable convolution: https://blogs.mathworks.com/steve/2006/10/04/separable-convolution/
   e = -sum_convolve(partial, ksize)

   # normalize
   return normalized_image(e)
   

# sum convolve with a kernel of size ksize
# example convolution kernel: [(1,1,1), (1,1,1), (1,1,1)]
@timeit
def sum_convolve(image, ksize):
   d = math.floor((ksize / 2))
   k = np.ones((d, d))
   return ndimage.convolve(image, k, mode='constant', cval=0.0)


@timeit
def normalized_image(image):
   """Normalize values of an `image` to values between 0 and 255, where
   The smallest value becomes 0 and the largest value becomes 255.
   
   Parameters
   ---------- 
   image : np.array (image)

   Returns
   ----------
   np.array
      A numpy array representing the normalized image.

   """

   image *= 255.0/image.max() 
   return image.astype(dtype=np.uint8)

# create an image where each pixel is the probability of that pixel relative 
# to the rest of the image
def pixel_probabilities(image):
   pass

# create an image where each pixel is the addends of the shannon entropy
# relative to the rest of the image
@timeit
def shannon_units(image, pdf):
   # mpdf = np.multiply(pdf, np.log2(pdf))
   mpdf = pdf * np.log2(pdf)
   return mpdf[image]
   #f = lambda v: mpdf[v]
   #f = np.vectorize(f, otypes=[np.float])
   #return f(image)

# calculate entropy of image
def entropy(image):

   # we convert 0 entries to 1 so that log2(pdf) evaluates
   # to 0 ranter than -inf (undefined). 
   pdf = probability_distribution(image, min(image.size, 256))
   return -np.sum(pdf * np.log2(pdf))

# calc entropy from a pdf
# @timeit
def ent(pdf):
   return -np.sum(pdf * np.log2(pdf))

# normalized probability distribution of grayscale pixel colors
@timeit
def probability_distribution(image, bins = 256, zero2hero = True):
   hist, bins = np.histogram(image.ravel(), 256, [0, 256])
   pd = hist / np.size(image)
   if zero2hero: pd[pd == 0] = 1;

   return pd

# calculate cross-entropy between two regions
# NOTE: either not working or not useful
def cross_entropy(file1, file2):
   pd1 = probability_distribution(file1)
   pd2 = probability_distribution(file2)

   lg = np.log2(pd1)
   # print("log2(pd1):")
   # print(lg)
   e = -np.sum(pd2 * lg)
   return e

# calculate entropy using convolution
# def entropy_filter(image, ksize):

# helper method to get a kernel region given an index
def kernel_region(image, yi, xi, ksize):
   height, width = image.shape[:2]
   offset = math.floor(ksize / 2)
   yStart = max(0, yi - offset)
   yEnd = min(height - 1, yi + offset)
   xStart = max(0, xi - offset)
   xEnd = min(width - 1, xi + offset)
   return image[yStart:yEnd, xStart:xEnd]

# generates an entropy map
def entropy_map(image, fsize, step):
   return filter_map(image, fsize, step, entropy)

# entropy map with fine focus
def entropy_map_fine(image):
   return entropy_map(image, 5, 2)

# entropy map with moderate focus
def entropy_map_mid(image):
   return entropy_map(image, 10, 5)

# entropy map with broad focus
def entropy_map_broad(image):
   return entropy_map(image, 50, 10)

# create a new map according to some filter f
def filter_map(image, fsize, step, f, normalize=True):
   # fmm = fmm(image, fsize, step)
   fMin = None
   fMax = None

   dims = focus_dimensions(image, fsize, step)
   fMap = np.zeros(dims)
   # print(fMap.shape)
   def process(x, y, focus):
      nonlocal fMin, fMax
      value = f(focus)
      # normalized = (value - fMin) / mDist
      fMap[int(y/step), int(x/step)] = value

      # check for min/max
      if (fMin is None) or (value < fMin):
         fMin = value
      if (fMax is None) or (value > fMax):
         fMax = value

   # scan the image
   focus_scan(image, fsize, step, process)

   # normalize the map
   if(normalize):
      fDist = fMax - fMin
      # print("fDist: %f"%(fDist))
      nMap = (fMap - fMin) / fDist
   
   return nMap

# DEPRECATED
# find region of lowest entropy
def lowest_entropy_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, False, entropy)

# DEPRECATED
# find region of highest entropy
def highest_entropy_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, True, entropy)

# DEPRECATED
# get min/max entropies
def mm_entropy(image, fsize, step):
   fMin = lowest_entropy_region(image, fsize, step)[2]
   fMax = highest_entropy_region(image, fsize, step)[2]
   print("eMin: %f, eMax: %f"%(fMin, fMax))
   return [fMin, fMax]

######################################################################
# Symmetry Functions
######################################################################

# calculate symmetry by comparing tiles along each strip
def symmetry_tile_map(image, size):
   
   vMap = symmetry_tile_vertical_map(image, 0, size)
   hMap = symmetry_tile_horizontal_map(image, 0, size)

   sMin = min(np.amin(vMap), np.amin(hMap))
   sMax = max(np.amax(vMap), np.amax(hMap))
   sDist = sMax - sMin

   sMap = np.zeros([vMap.shape[0], vMap.shape[1]])
   sMap = (((vMap - sMin) / sDist) + ((hMap - sMin) / sDist)) / 2

   # normalize
   # print("sMin: %f, sMax: %f, sDist: %f"%(sMin, sMax, sDist))
   # if sDist != 0:
   #    sMap = (sMap - sMin) / sDist

   # print(sMap)
   return sMap
   # vFrame = symmetry_frame(image, 1, size)
   # hFrame = symmetry_frame(image, 0, size)
   # numRows = vFrame[0]
   # numCols = hFrame[0]

   # for y in range(0, numRows):
   #    for x in range(0, numCols):

def symmetry_tile_vertical_map(image, axis, size):
   height, width = image.shape[:2]
   vFrame = symmetry_frame(image, 1, size)
   hFrame = symmetry_frame(image, 0, size)
   numRows = vFrame[0]
   numCols = hFrame[0]
   offsetY = vFrame[1]
   offsetX = hFrame[1]

   sMap = np.zeros([numRows, numCols])
   # print(sMap.shape)
   for x in range(0, numCols):
      for y in range(0, int(numRows / 2)):
         x1 = offsetX + (x * size)
         x2 = x1 + size
         yt1 = offsetY + (y * size)
         yt2 = yt1 + size
         yb1 = height - offsetY - ((y+1) * size)
         yb2 = height - offsetY - (y * size)

         slice1 = image[yt1:yt2, x1:x2]
         slice2 = image[yb1:yb2, x1:x2]

         sym = np.average(abs(slice1-slice2))
         sMap[y, x] = sym
         sMap[numRows - 1 - y, x] = sym

         # print('x: %d, y1: %d, y2: %d'%(x, y, numRows - 1 - y))

   # normalize
   # sMin = np.amin(sMap)
   # sMax = np.amax(sMap)
   # sDist = sMax - sMin
   # print("sMin: %f, sMax: %f, sDist: %f"%(sMin, sMax, sDist))
   # if sDist != 0:
   #    sMap = (sMap - sMin) / sDist
   
   # print(sMap)

   return sMap

def symmetry_tile_horizontal_map(image, axis, size):
   height, width = image.shape[:2]
   vFrame = symmetry_frame(image, 1, size)
   hFrame = symmetry_frame(image, 0, size)
   numRows = vFrame[0]
   numCols = hFrame[0]
   offsetY = vFrame[1]
   offsetX = hFrame[1]

   # print('numRows: %d, numCols: %d'%(numRows, numCols))
   sMap = np.zeros([numRows, numCols])
   # print(sMap.shape)
   for y in range(0, numRows):
      for x in range(0, int(numCols / 2)):
         y1 = offsetY + (y * size)
         y2 = y1 + size
         xl1 = offsetX + (x * size)
         xl2 = xl1 + size
         xr1 = width - offsetX - ((x+1) * size)
         xr2 = width - offsetX - (x * size)

         # print('xl1: %d, xl2: %d, xr1: %d, xr2: %d'%(xl1, xl2, xr1, xr2))

         slice1 = image[y1:y2, xl1:xl2]
         slice2 = image[y1:y2, xr1:xr2]

         sym = np.average(abs(slice1-slice2))
         sMap[y, x] = sym
         sMap[y, numCols - 1 - x] = sym

         # print('x: %d, y1: %d, y2: %d'%(x, y, numRows - 1 - y))

   # normalize
   # sMin = np.amin(sMap)
   # sMax = np.amax(sMap)
   # sDist = sMax - sMin
   # print("sMin: %f, sMax: %f, sDist: %f"%(sMin, sMax, sDist))
   # if sDist != 0:
   #    sMap = (sMap - sMin) / sDist
   
   # print(sMap)

   return sMap

# calculate symmetry by comparing symmetry of full width/height slices
# across both axis. symmetry from both axis is combined by addition 
# and normalization
def symmetry_strip_map(image, size):
   vSymmetry = symmetry_strip_axis_map(image, 1, size)[1]
   hSymmetry = symmetry_strip_axis_map(image, 0, size)[1]
   
   sMap = np.zeros([2*vSymmetry.size, 2*hSymmetry.size])
   # sMap = np.zeros([vSymmetry.size, hSymmetry.size])

   sMin = min(np.amin(vSymmetry), np.amin(hSymmetry))
   sMax = max(np.amax(vSymmetry), np.amax(hSymmetry))
   sDist = sMax - sMin

   for y in range(0, vSymmetry.size):
      for x in range(0, hSymmetry.size):
         y2 = 2*vSymmetry.size - y - 1
         x2 = 2*hSymmetry.size - x - 1

         symmetry = ((vSymmetry[y] - sMin) / sDist) + ((hSymmetry[x] - sMin) / sDist)
         symmetry /= 2
         # print(symmetry)
         sMap[y,x] = symmetry
         sMap[y2,x2] = symmetry
         sMap[y,x2] = symmetry
         sMap[y2,x] = symmetry

         # # check for min/max
         # if (sMin is None) or (symmetry < sMin):
         #    sMin = symmetry
         # if (sMax is None) or (symmetry > sMin):
         #    sMax = symmetry

   # normalize
   # sDist = sMax - sMin
   # print("sMin: %f, sMax: %f, sDist: %f"%(sMin, sMax, sDist))
   # sMap = (sMap - sMin) / sDist
   
   return sMap


# calculate symmetry for an image along either vertical or horz axis
def symmetry_strip_axis_map(image, axis, size):
   frame = symmetry_frame(image, axis, size)
   numRegions = frame[0]
   offset = frame[1]
   height, width = image.shape[:2]

   # print("size: %d, axis: %d, numRegions: %d, offset: %d"%(size, axis, numRegions, offset))
   # symmetry across x (vertical)
   
   region = height if (axis == 1) else width

   # sMax = None
   # sMin = None
   smap = np.zeros(int(numRegions / 2))

   for i in range(0, int(numRegions / 2)):
      coords = symmetry_coords(i, region, offset, size)

      if(axis == 1):
         slice1 = image[coords[0][0]:coords[0][1], :]
         slice2 = image[coords[1][0]:coords[1][1], :]
      else:
         slice1 = image[:, coords[0][0]:coords[0][1]]
         slice2 = image[:, coords[1][0]:coords[1][1]]

      symmetry = np.average(abs(slice1-slice2))
      smap[i] = symmetry

      # # check for min/max
      # if (sMin is None) or (symmetry < sMin):
      #    sMin = symmetry
      # if (sMax is None) or (symmetry > sMin):
      #    sMax = symmetry


   # normalize
   # sDist = sMax - sMin
   # smap = (smap - sMin) / sDist
   
   return [axis, smap]

def symmetry_frame(image, axis, size):
   height, width = image.shape[:2]

   numRegions = (height / size) if (axis == 1) else (width / size)
   numRegions = math.floor(numRegions)
   if(numRegions % 2): numRegions -= 1

   offset = (height - (size * numRegions)) if (axis == 1) else (width - (size * numRegions))
   offset = math.floor(offset / 2)

   return [numRegions, offset]

def symmetry_coords(index, frame, offset, size):
   start1 = offset+(index*size)
   end1 = offset+((index+1)*size)
   start2 = frame - offset - 1 - ((index+1)*size)
   end2 = frame - 1 - offset - (index * size)
   return [[start1, end1], [start2, end2]]

######################################################################
# Misc Functions
######################################################################

# sobel filter
def sobel_filter(image, ksize):
   sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
   scalex = cv2.convertScaleAbs(sobelx)
   sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
   scaley = cv2.convertScaleAbs(sobelx)
   scaled = cv2.addWeighted(scalex, 0.5, scaley, 0.5, 0)
   return scaled

# get dimensions of output array from a focus scan
def focus_dimensions(image, fsize, step):
   height, width = image.shape[:2]
   cols = int(math.floor((width - fsize) / step))
   rows = int(math.floor((height - fsize) / step))
   return [rows, cols]

# scan an image, focusing on regions of fsize x fsize, stepping every
# step pixels, callback function for processing
def focus_scan(image, fsize, step, callback):
   dims = focus_dimensions(image, fsize, step)

   for yi in range(0, dims[0]):
      for xi in range(0, dims[1]):
         y = yi * step
         x = xi * step
         focus = image[y:y+fsize, x:x+fsize]
         callback(x, y, focus)

# scan image for a min or max value according to some function f
# f can be entropy(image) or contrast(image) 
def mm_region_scan(image, fsize, step, mm, f):
   extreme = None
   r = [0, 0]

   # check for the lowest entropy
   def process(x, y, focus):
      nonlocal extreme, r

      val = f(focus)
      if extreme is None: extreme = val 
      if ((not mm) and (val < extreme)) or (mm and (val > extreme)):
         extreme = val
         r = [x, y]

   # scan the image
   focus_scan(image, fsize, step, process)

   return [r, fsize, extreme]

######################################################################
# Color channels
######################################################################

# extact color channel
def extract_color_channel(image, kernel):
   # np.dot(image, np.array([1,1,1]))
   return image[:,:,kernel]
   
# extract blue channel
def extract_blue(image):
   return extract_color_channel(image, 0)

# extract green channel
def extract_green(image):
   return extract_color_channel(image, 1)

# extract red channel
def extract_red(image):
   return extract_color_channel(image, 2)

######################################################################
# Contrast Functions (unused)
######################################################################

def contrast(image):
   pd = probability_distribution(image, False)
   lower, upper = np.split(pd, 2)
   halfSize = len(lower)

   # weighted overage of lower half
   indices = np.arange(0, halfSize)
   lAvg = np.average(indices, weights = lower)

   # weighted overage of upper half
   uAvg = 0
   if(np.sum(upper) > 0):
      indices = np.arange(halfSize, 2 * halfSize)
      uAvg = np.average(indices, weights = upper)

   return (uAvg - lAvg) / (2 * halfSize)

# find region of lowest contrast
def lowest_contrast_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, False, contrast)

# find region of highest contrast
def highest_contrast_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, True, contrast)

# get min/max contrasts
def mm_contrast(image, fsize, step):
   fMin = lowest_contrast_region(image, fsize, step)[2]
   fMax = highest_contrast_region(image, fsize, step)[2]
   return [fMin, fMax]

# NOTE: contrast less useful than entropy - see the leopard image for example
# generates a contrast map
def contrast_map(image, fsize, step):
  return filter_map(image, fsize, step, contrast)

# NOTE: does not seem useful
# generates a hybrid entropy/contrast map
def hybrid_map(image, fsize, step):
   cmap = contrast_map(image, fsize, step)
   emap = entropy_map(image, fsize, step)
   hmap = np.add(cmap, emap)
   return np.divide(hmap, 2)

########################################################################
# Helper functions
########################################################################

# replace elements with the number of times they occur in a window
# returns an array of shape (a.height, a.width, ksize, ksize),
# with each ksize x ksize window containing counts of how many times that pixel occurred
def windowed_occurences(a, ksize):
   window_shape = (ksize, ksize)
   d = math.floor(ksize / 2)
   a = np.pad(a, (d, d), 'constant', constant_values=(0, 0)) # constant_values=(-1, -1)

   # get the windowed array of shape (a.height, a.width, ksize, ksize)
   b = view_as_windows(a, window_shape)

   # replace each element of ksize x ksize kernel with the number of occurances
   # of that element in the kernel
   ar2D = b.reshape(-1, b.shape[-2] * b.shape[-1])

   #c = bincount2D_numba(ar2D, use_parallel=True, use_prange=True)
   c = bincount2D_vectorized(ar2D)
   return c[np.arange(ar2D.shape[0])[:, None], ar2D].reshape(b.shape)  

# Helper functions for vectorizing bincount
# https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy/46256361#46256361

# Vectorized solution (simple but non-performant)
def bincount2D_vectorized(a):
   print("bincount2D_vectorized:")
   N = a.max() + 1
   a_offs = a + np.arange(a.shape[0])[:, None] * N
   return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1, N)

# Numba solutions
def bincount2D_numba(a, use_parallel=False, use_prange=False):
    N = a.max()+1
    m,n = a.shape
    out = np.zeros((m,N),dtype=int)

    # Choose fucntion based on args
    func = bincount2D_numba_func0
    if use_parallel:
        if use_prange:
            func = bincount2D_numba_func2
        else:
            func = bincount2D_numba_func1
    # Run chosen function on input data and output
    func(a, out, m, n)
    return out

@njit
def bincount2D_numba_func0(a, out, m, n):
    for i in range(m):
        for j in range(n):
            out[i,a[i,j]] += 1

@njit(parallel=True)
def bincount2D_numba_func1(a, out, m, n):
    for i in range(m):
        for j in range(n):
            out[i,a[i,j]] += 1

@njit(parallel=True)
def bincount2D_numba_func2(a, out, m, n):
    for i in prange(m):
        for j in prange(n):
            out[i,a[i,j]] += 1

########################################################################
# Deprecated/Unused functions
########################################################################

@timeit
def entropy_local_old(image, ksize = 9):
   if (not ksize % 2) or ksize < 3:
      raise Exception("Kernel must be of odd size > 1")

   # THIS IS AN INTERESTING APPROACH AND MAY STILL BE USEFUL
   # # get the number of times a value occurs in each kernel window
   # a = windowed_occurences(image, ksize)

   # # convert to probabilities
   # p = a / (ksize * ksize)

   # # get the shannon units
   # s = p * np.log2(p)
   
   # # (technically this sum is inaccurate, since values that appear 
   # # more than once get summed multiple times)
   # # sum them to calculate entropy
   # e = s.sum((-2, -1))
   # print(e)

   @timeit
   def calc_e(*args, **kwargs):
      height, width = image.shape[:2]   
      eMap = np.zeros([height, width])
      for yi in range(0, height):
         for xi in range(0, width):
            region = kernel_region(image, yi, xi, ksize)
            e = entropy(region)
            eMap[yi, xi] = e

      return eMap
   eMap = calc_e()
   #print(eMap)

   return normalized_image(eMap)

# some ad-hoc box filter
def box_filter(image, res):
   sqr = res*res
   height, width = image.shape[:2]
   print("averaging: (%d, %d)"%(width, height))
   blank_image = np.zeros((height,width,3), np.uint8)
   # blank_image[:,0:int(0.5*width)] = (255,0,0)      # (B, G, R)
   # blank_image[:,int(0.5*width):width] = (0,255,0)

   for i in range(0, int(math.floor((height)/res))):
      for j in range(0, int(math.floor((width)/res))):
         y = res * i
         x = res * j
         avg = (0, 0, 0)
         for k in range(0, res):
            for l in range(0, res):
               rgb = image[y+k, x+l]
               avg += rgb
               #print("(%d, %d)"%(x+l, y+k))

         avg = avg / (sqr, sqr, sqr)
         # update image
         blank_image[y:(y+res),x:(x+res)] = avg

   return blank_image

# find areas of entropy close to one
def line_finder(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   cv2.imshow('win1', image)
   cv2.waitKey(0)

   fsize = 5 # size of focus region
   threshold = 0.5
   detect = 1
   height, width = image.shape[:2]

   highEntropyPoints = []
   for y in range(0, height - fsize):
      for x in range(0, width - fsize):
         focus = image[y:y+fsize, x:x+fsize]
         fentropy = entropy(focus)
         if abs(detect - fentropy) < threshold:
            highEntropyPoints.append([x, y])

   for i in range(0, len(highEntropyPoints)):
      x = highEntropyPoints[i][0];
      y = highEntropyPoints[i][1];
      colored_image[y:(y+fsize),x:(x+fsize)] = (0, 255, 0)
   # print("done")
   
   cv2.imshow('win1', colored_image)
   cv2.waitKey(0)

# def filter_map(image, fsize, step, f, fmm):
#    fmm = fmm(image, fsize, step)
#    fMin = fmm[0]
#    fMax = fmm[1]
#    mDist = fMax - fMin

#    fMap = []
#    print("fMin: %f, fMax: %f"%(fMin, fMax))
#    def process(x, y, focus):
#       nonlocal fMin, mDist
#       value = f(focus)
#       normalized = (value - fMin) / mDist
#       fMap.append((x, y, normalized))

#    # scan the image
#    focus_scan(image, fsize, step, process)

#    return fMap

# # extact color channel
# def extract_color_channel(image, kernel):
#    # leaves the array in full-color
#    channel = np.copy(image)
#    height, width = image.shape[:2]
#    for y in range(0, height):
#       for x in range(0, width):
#          channel[y,x] = image[y, x] * kernel
#    return channel # cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)