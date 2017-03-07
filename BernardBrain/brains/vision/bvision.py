import numpy as np
import cv2
import time
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

######################################################################
# Entropy Functions
######################################################################

# calculate entropy of image
def entropy(image):
   pdf = probability_distribution(image);
   e = -np.sum(pdf * np.log2(pdf))
   return e

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

# normalized probability distribution of grayscale pixel colors
def probability_distribution(image, zero2hero = True):
   hist, bins = np.histogram(image.ravel(), 256, [0, 256])
   pd = hist / np.size(image);
   if zero2hero: pd[pd == 0] = 1;

   return pd

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

######################################################################
# Old & Unused
######################################################################

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