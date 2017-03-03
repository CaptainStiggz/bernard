import numpy as np
import cv2
import time
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import bvision as bv
import bvisionui as bui
from os import listdir
from os.path import isfile, join

######################################################################
# Current Tests
######################################################################

# run a test battery on all files in BSD test until user quits
def test_battery(f):
   folder = 'BSDS300/images/train'
   files = [f for f in listdir(folder) if isfile(join(folder, f))]
   for file in files:
      file = folder+'/'+file
      print(file)
      f(file)

def symmetry_test(file):
   image = cv2.imread(file, 0)
   vsym = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   hsym = np.copy(vsym)
   tsym = np.copy(vsym)

   size = 20
   smap = bv.symmetry_map(image, size)
   vmap = bv.symmetry_axis_map(image, 1, size)
   hmap = bv.symmetry_axis_map(image, 0, size)

   bui.paint_symmetry(tsym, smap, size)
   bui.paint_axis_symmetry(vsym, vmap, size)
   bui.paint_axis_symmetry(hsym, hmap, size)

   bui.show_images([image, vsym, hsym, tsym])

# extract color channels and display results
def color_extraction_test(file):
   image = cv2.imread(file, cv2.IMREAD_COLOR)
   # print(image.shape)
   # print(np.multiply(image, (1, 1, 1)).shape)
   # image = map(lambda x: x * (1, 1, 1), image)
   #image = np.arange(5*5*3).reshape(5,5,3)
   #image = np.ones([5, 5, 3]) * 255
   bui.show_images([image, bv.extract_blue(image), bv.extract_green(image), bv.extract_red(image)])

# create a map on the image, highest entropy in red, lowest in green
def entropy_map_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   bui.show_images([image])

   fsize = 50
   step = 10
   fsize = 10
   step = 5
   rsize = step

   emap = bv.entropy_map(image, fsize, step)
   bui.paint_map(colored_image, emap, fsize, step, rsize)
   bui.show_images([image, colored_image])

# displays entropy of each color channel individually
def color_channel_entropy_test(file):
   image = cv2.imread(file, cv2.IMREAD_COLOR)
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blues = bv.extract_blue(image)
   greens = bv.extract_green(image)
   reds = bv.extract_red(image)
   blueMap = np.copy(image)
   redMap = np.copy(image)
   greenMap = np.copy(image)
   grayMap = np.copy(image)

   bui.show_images([blues, greens, reds])

   fsize = 50
   step = 10
   # fsize = 8
   # step = 4
   rsize = step

   print("eGray: %f eBlue: %f, eGreen: %f, eRed: %f"%(bv.entropy(gray), bv.entropy(blues), bv.entropy(greens), bv.entropy(reds)))
   emap = bv.entropy_map(gray, fsize, step)
   bmap = bv.entropy_map(blues, fsize, step)
   gmap = bv.entropy_map(greens, fsize, step)
   rmap = bv.entropy_map(reds, fsize, step)

   bui.paint_map(grayMap, bmap, fsize, step, rsize)
   bui.paint_map(blueMap, bmap, fsize, step, rsize)
   bui.paint_map(greenMap, gmap, fsize, step, rsize)
   bui.paint_map(redMap, rmap, fsize, step, rsize)
   bui.show_images([image, gray, grayMap, blues, greens, reds, blueMap, redMap, greenMap])

# displays entropy of all color channels, combined
def colored_entropy_test(file):
   image = cv2.imread(file, cv2.IMREAD_COLOR)
   gray = cv2.imread(file, 0)
   #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blues = extract_blue(image)
   greens = extract_green(image)
   reds = extract_red(image)
   map_image = np.copy(image)

   show_images([image, gray, blues, greens, reds])

   fsize = 50
   step = 10
   # fsize = 10
   # step = 5
   rsize = step

   emap = entropy_map(gray, fsize, step)
   bmap = entropy_map(blues, fsize, step)
   gmap = entropy_map(greens, fsize, step)
   rmap = entropy_map(reds, fsize, step)

   height, width = emap.shape[:2]
   fmap = np.copy(emap)
   for y in range(0, height):
      for x in range(0, width):
         fmap[y,x] = (emap[y, x] + bmap[y, x] + gmap[y, x] + rmap[y, x]) / 4

   paint_map(map_image, fmap, fsize, step, rsize)
   show_images([image, gray, blues, greens, reds, map_image])

######################################################################
# Old Tests
######################################################################

# calculate contrast of image
def contrast_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   # show_images([image])

   c = contrast(image)
   print("contrast %f"%(c))
   show_images([image])

def super_map_test(file):
   image = cv2.imread(file, 0)
   eimage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   himage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10
   rsize = 6
   # fsize = 20
   # step = 5

   cmap = contrast_map(image, fsize, step)
   emap = entropy_map(image, fsize, step)
   hmap = hybrid_map(image, fsize, step)
   
   paint_map(cimage, cmap, fsize, step,rsize)
   paint_map(eimage, emap, fsize, step, rsize)
   paint_map(himage, hmap, fsize, step, rsize)
   show_images([image, cimage, eimage, himage])

# create a map on the image, highest entropy in red, lowest in green
def hybrid_map_test(file):
   image = cv2.imread(file, 0)
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   show_images([image])

   fsize = 50
   step = 10
   rsize = 6
   # fsize = 20
   # step = 5

   cmap = hybrid_map(image, fsize, step)
   paint_map(colored_image, cmap, fsize, step, rsize)
   show_images([image, colored_image])

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

   cmap = contrast_map(image, fsize, step)
   paint_map(colored_image, cmap, fsize, step, rsize)
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

# doesn't seem very helpful
def cross_entropy_test(file1, file2):
   print("\nENTROPY TEST: %s, %s\n"%(file1, file2))
   image1 = cv2.imread(file1, 0)
   image2 = cv2.imread(file1, 0)
   cv2.imshow('win1', image1)
   cv2.imshow('win2', image2)
   cv2.moveWindow('win2', 100+image1.shape[1], 100);

   print("entropy(image1): %f, entropy(image2): %f"%(entropy(image1), entropy(image2)))
   print("cross entropy 1x2: %f, cross entropy 2x1: %f"%(cross_entropy(image1, image2), cross_entropy(image2, image1)))
   cv2.waitKey(0)
   print("\nDONE\n")

def cross_entropy_test_battery():
   cross_entropy_test('entropy_images/black.jpg', 'entropy_images/black.jpg')
   cross_entropy_test('entropy_images/white.jpg', 'entropy_images/white.jpg')
   cross_entropy_test('entropy_images/black.jpg', 'entropy_images/white.jpg')
   cross_entropy_test('entropy_images/noise.jpg', 'entropy_images/noise.jpg')
   cross_entropy_test('entropy_images/noise.jpg', 'entropy_images/black.jpg')
   cross_entropy_test('entropy_images/noise.jpg', 'entropy_images/white.jpg')

   cross_entropy_test('entropy_images/bwv.jpg', 'entropy_images/bwv.jpg')
   cross_entropy_test('entropy_images/bwv.jpg', 'entropy_images/wbv.jpg')

   cross_entropy_test('entropy_images/bwh.jpg', 'entropy_images/bwh.jpg')
   cross_entropy_test('entropy_images/bwv.jpg', 'entropy_images/wbv.jpg')

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

file = 'entropy_images/bwv.jpg'
# file = 'images/symmetrytest.jpg'
# file = 'bgrtest2.jpg' # color test
# file = 'BSDS300/images/train/87065.jpg' # lizard
# file = 'BSDS300/images/train/100075.jpg' # bears
# file = 'BSDS300/images/train/134052.jpg' # leopard
# file = 'entropygirl.png'
# file = 'leopard-ecrop.jpg' # leopard
# file = 'BSDS300/images/train/181018.jpg' # some girl
# file = 'BSDS300/images/train/15004.jpg' # lady in the market

test_battery(symmetry_test)
# symmetry_test(file)
# test_battery(color_channel_entropy_test)
# color_channel_entropy_test(file)
# colored_entropy_test(file)
# color_extraction_test(file)
# line_finder(file)
# super_map_test(file)
# hybrid_map_test(file)
# contrast_map_test(file)
# entropy_map_test(file)
# cross_entropy_test_battery()



cv2.destroyAllWindows()