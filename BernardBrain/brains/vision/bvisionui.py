import os
import pathlib
import numpy as np
import cv2
import time
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from .bvision import *

########################################################################
# UI
########################################################################

def plot_images(images, max_cols=3, norm=None):
   """Plot a list of images using matplotlib.

   Parameters
   ---------- 
   images : list : np.array (image)
   max_cols : int
      The maximum number of columns to use.

   """

   cv2.destroyAllWindows()
   fig = plt.figure()
   #plt.axis("off")

   cols = max(1, min(len(images), max_cols))
   rows = max(1, math.ceil(len(images) / max_cols))
   print("rows: %d cols: %d"%(rows, cols))
   for i in range(0, len(images)):
      ax = fig.add_subplot(rows, cols, i+1)
      #ax.axis("off")
      ax.imshow(images[i], cmap='gray', norm=norm)

   plt.show()

def extract_video_frames(filename):
   """Extract frames of a video file as individual images.

   Parameters
   ---------- 
   filename : string
      The relative path of the input file.

   Returns
   ----------
   boolean
      True if successful, false otherwise.

   """

   source = pathlib.Path(filename)
   if source.is_file():

      # Create output directory.
      dest = filename+"-frames"
      pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

      # Read video frames.
      vidcap = cv2.VideoCapture(filename)
      success,image = vidcap.read()
      count = 0
      success = True
      while success:
         cv2.imwrite(dest+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
         success,image = vidcap.read()
         count += 1

      return True
   else:
      raise Exception("Soure file: "+filename+" does not exist.")

   return False

########################################################################
# Deprecated
########################################################################

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
      cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
      cv2.moveWindow(windowName, origin[0]+shiftX, origin[1]+shiftY)
      cv2.imshow(windowName, image)

      # print("w: %d, h: %d, sx: %d, sy: %d"%(width, height, shiftX, shiftY))

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

def paint_map(image, fMap, fsize, step, rsize):
   for (y,x), value in np.ndenumerate(fMap):
   # for element in fMap:
      x *= step
      y *= step
      fRed = value
      fGreen = 1 - fRed

      color = (0, int(fGreen * 255), int(fRed * 255))
      offset = (fsize - rsize) / 2
      rect = [int(x + offset), int(y + offset), rsize, rsize]
      fill_rect(image, rect, color)

def paint_tile_symmetry(image, sMap, size):
   height, width = image.shape[:2]

   offsetX = bv.symmetry_frame(image, 0, size)[1]
   offsetY = bv.symmetry_frame(image, 1, size)[1]

   for (y,x), value in np.ndenumerate(sMap):
      fGreen = value
      fRed = 1 - fGreen
      color = (0, int(fGreen * 255), int(fRed * 255))
      rect = [int(x * size + offsetX), int(y * size + offsetY), size, size]
      fill_rect(image, rect, color)

def paint_symmetry(image, sMap, size):
   height, width = image.shape[:2]

   offsetX = bv.symmetry_frame(image, 0, size)[1]
   offsetY = bv.symmetry_frame(image, 1, size)[1]

   for (y,x), value in np.ndenumerate(sMap):
      fGreen = value
      fRed = 1 - fGreen
      color = (0, int(fGreen * 255), int(fRed * 255))
      rect = [int(x * size + offsetX), int(y * size + offsetY), size, size]
      fill_rect(image, rect, color)

def paint_axis_symmetry(image, sMap, size):
   axis = sMap[0]
   sym = sMap[1]
   frame = bv.symmetry_frame(image, axis, size)
   numRegions = frame[0]
   offset = frame[1]
   height, width = image.shape[:2]
   region = height if (axis == 1) else width

   for i in range(0, int(numRegions / 2)):
      value = sym[i]
      fGreen = value
      fRed = 1 - fGreen
      color = (0, int(fGreen * 255), int(fRed * 255))

      coords = bv.symmetry_coords(i, region, offset, size)
      if axis == 1:
         image[coords[0][0]:coords[0][1], :] = color
         image[coords[1][0]:coords[1][1], :] = color
      else:
         image[:, coords[0][0]:coords[0][1]] = color
         image[:, coords[1][0]:coords[1][1]] = color



