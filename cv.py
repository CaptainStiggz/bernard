import numpy as np
import cv2
import time
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

######################################################################
# UI
######################################################################

def show_images(images):
   cv2.destroyAllWindows()
   origin = (100, 100)
   
   shiftX = 0
   shiftY = 0
   for i in range(0, len(images)):
      image = images[i]

      height, width = image.shape[:2]
      shiftX = (i%3) * width
      shiftY = math.floor(i/3) * height

      windowName = 'win%d'%(i)
      cv2.namedWindow(windowName)
      cv2.moveWindow(windowName, origin[0]+shiftX, origin[1]+shiftY)
      cv2.imshow(windowName, image)

      print("w: %d, h: %d, sx: %d, sy: %d"%(width, height, shiftX, shiftY))

   cv2.waitKey(0)

# draw a box around some rectangle (not a very safe method, might crash)
def outline_rect(image, rect, color):
   x = rect[0]
   y = rect[1]
   w = rect[2]
   h = rect[3]

   image[y:(y+1),x:x+w] = color # top
   image[y:(y+h),x:x+1] = color # left
   image[y+h-1:(y+h),x:x+w] = color # bottom
   image[y:(y+h),x+w-1:x+w] = color # right

def fill_rect(image, rect, color):
   x = rect[0]
   y = rect[1]
   w = rect[2]
   h = rect[3]

   image[y:y+h,x:x+w] = color

######################################################################
# Filters and image processing
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

def disorder_between(image, r1, r2):
   pass

# normalized probability distribution of grayscale pixel colors
def probability_distribution(image, zero2hero = True):
   hist, bins = np.histogram(image.ravel(), 256, [0, 256])
   pd = hist / np.size(image);
   if zero2hero: pd[pd == 0] = 1;
   return pd

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

def contrast(image):
   pd = probability_distribution(image, False)
   lower, upper = np.split(pd, 2)
   halfSize = len(lower)

   # weighted overage of lower half
   indices = np.arange(0, halfSize)
   lAvg = np.average(indices, weights = lower)

   # weighted overage of upper half
   indices = np.arange(halfSize, 2 * halfSize)
   uAvg = np.average(indices, weights = upper)

   return (uAvg - lAvg) / (2 * halfSize)

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


# find region of lowest contrast
def lowest_contrast_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, False, contrast)

# find region of highest contrast
def highest_contrast_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, True, contrast)

# find region of lowest entropy
def lowest_entropy_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, False, entropy)

# find region of highest entropy
def highest_entropy_region(image, fsize, step):
   return mm_region_scan(image, fsize, step, True, entropy)

def focus_scan(image, fsize, step, callback):
   height, width = image.shape[:2]
   cols = int(math.floor((width - fsize) / step))
   rows = int(math.floor((height - fsize) / step))

   for yi in range(0, rows):
      for xi in range(0, cols):
         y = yi * step
         x = xi * step
         focus = image[y:y+fsize, x:x+fsize]
         callback(x, y, focus)

######################################################################
# Tests & runnables
######################################################################

# calculate contrast of image
def contrast_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   # show_images([image])

   c = contrast(image)
   print("contrast %f"%(c))
   show_images([image])

# create a map on the image, highest entropy in red, lowest in green
def contrast_map_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10
   rsize = 6
   # fsize = 20
   # step = 5

   eMin = lowest_contrast_region(image, fsize, step)[2]
   eMax = highest_contrast_region(image, fsize, step)[2]

   print("cMin: %f, cMax: %f"%(eMin, eMax))
   def process(x, y, focus):
      nonlocal eMin, eMax
      fentropy = contrast(focus)

      mDist = eMax - eMin
      fRed = (fentropy - eMin) / mDist
      fGreen = (eMax - fentropy) / mDist

      color = (0, int(fGreen * 255), int(fRed * 255))

      offset = (fsize - rsize) / 2
      rect = [int(x + offset), int(y + offset), rsize, rsize]
      fill_rect(colored_image, rect, color)

   # scan the image
   focus_scan(image, fsize, step, process)

   show_images([image, colored_image])

# create a map on the image, highest entropy in red, lowest in green
def entropy_map_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10
   rsize = 6
   # fsize = 20
   # step = 5

   eMin = lowest_entropy_region(image, fsize, step)[2]
   eMax = highest_entropy_region(image, fsize, step)[2]

   print("eMin: %f, eMax: %f"%(eMin, eMax))
   def process(x, y, focus):
      nonlocal eMin, eMax
      fentropy = entropy(focus)

      mDist = eMax - eMin
      fRed = (fentropy - eMin) / mDist
      fGreen = (eMax - fentropy) / mDist

      color = (0, int(fGreen * 255), int(fRed * 255))

      offset = (fsize - rsize) / 2
      rect = [int(x + offset), int(y + offset), rsize, rsize]
      fill_rect(colored_image, rect, color)

   # scan the image
   focus_scan(image, fsize, step, process)

   show_images([image, colored_image])

# draw boxes around the highest and lowest contrast regions
def contrast_extremes_test(file):
   image = cv2.imread(file, 0)
   low_entropy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   high_entropy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10

   # color lowest entropy green
   region = lowest_contrast_region(image, fsize, step)
   x = region[0][0];
   y = region[0][1];
   outline_rect(low_entropy_image, [x, y, fsize, fsize], (0, 255, 0))
   show_images([image, low_entropy_image])

   # color highest entropy red
   region = highest_contrast_region(image, fsize, step)
   x = region[0][0];
   y = region[0][1];
   outline_rect(high_entropy_image, [x, y, fsize, fsize], (0, 0, 255))
   show_images([image, low_entropy_image, high_entropy_image])

# draw boxes around the highest and lowest entropy regions
def entropy_extremes_test(file):
   image = cv2.imread(file, 0)
   low_entropy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   high_entropy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10

   # color lowest entropy green
   region = lowest_entropy_region(image, fsize, step)
   x = region[0][0];
   y = region[0][1];
   outline_rect(low_entropy_image, [x, y, fsize, fsize], (0, 255, 0))
   show_images([image, low_entropy_image])

   # color highest entropy red
   region = highest_entropy_region(image, fsize, step)
   x = region[0][0];
   y = region[0][1];
   outline_rect(high_entropy_image, [x, y, fsize, fsize], (0, 0, 255))
   show_images([image, low_entropy_image, high_entropy_image])

def cross_entropy_test(file1, file2):
   print("\nENTROPY TEST: %s, %s\n"%(file1, file2))
   image1 = cv2.imread("entropy_images/"+file1, 0)
   image2 = cv2.imread("entropy_images/"+file2, 0)
   cv2.imshow('win1', image1)
   cv2.imshow('win2', image2)
   cv2.moveWindow('win2', 100+image1.shape[1], 100);

   print("entropy(image1): %f, entropy(image2): %f"%(entropy(image1), entropy(image2)))
   print("cross entropy 1x2: %f, cross entropy 2x1: %f"%(cross_entropy(image1, image2), cross_entropy(image2, image1)))
   cv2.waitKey(0)
   print("\nDONE\n")

def cross_entropy_test_battery():
   cross_entropy_test('black.jpg', 'black.jpg')
   cross_entropy_test('white.jpg', 'white.jpg')
   cross_entropy_test('black.jpg', 'white.jpg')
   cross_entropy_test('noise.jpg', 'noise.jpg')
   cross_entropy_test('noise.jpg', 'black.jpg')
   cross_entropy_test('noise.jpg', 'white.jpg')

   cross_entropy_test('bwv.jpg', 'bwv.jpg')
   cross_entropy_test('bwv.jpg', 'wbv.jpg')

   cross_entropy_test('bwh.jpg', 'bwh.jpg')
   cross_entropy_test('bwv.jpg', 'wbv.jpg')

def test1(file):
   image = cv2.imread(file, cv2.IMREAD_COLOR)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   cv2.imshow('win1', image)
   cv2.imshow('win2', gray)
   cv2.moveWindow('win2', 100+image.shape[1], 100);
   # cv2.moveWindow('win3', 100+image.shape[1], 100+image.shape[0]);
   # plt.hist(np.histogram(gray.ravel(), 256, [0, 256])) 
   # plt.show()
   print("entropy: %f"%(entropy(gray)))
   cv2.waitKey(0)

######################################################################
# main
######################################################################

# file = 'entropy_images/bwv.jpg'
file = 'BSDS300/images/train/87065.jpg' # lizard
# file = 'BSDS300/images/train/134052.jpg' # leopard
# file = 'BSDS300/images/train/181018.jpg' # some girl
# file = 'BSDS300/images/train/15004.jpg' # lady in the market

# line_finder(file)

contrast_map_test(file)
# entropy_map_test(file)
# cross_entropy_test_battery()



cv2.destroyAllWindows()