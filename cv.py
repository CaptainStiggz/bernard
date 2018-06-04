import numpy as np
import cv2
import time
import math
import matplotlib as mpl
import skimage
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import BernardBrain.brains.vision.bvision as bv
import BernardBrain.brains.vision.bvisionui as bui
from os import listdir
from os.path import isfile, join

######################################################################
# Current Tests
######################################################################

def entropy_image(file):
   image = cv2.imread(file, 0)
   blur = cv2.GaussianBlur(image, (5,5), 0)

   # global_entropy = bv.entropy_global(blur, 9)
   # local_entropy = bv.entropy_local(blur, 3)
   # reduced_entropy = bv.entropy_local(blur, 61, 50)
   results = bv.entropy_stack(image)
   
   bui.plot_images([image] + results)
   # fig = plt.figure()

   # ax1 = fig.add_subplot(121)
   # ax1.imshow(local_entropy, cmap='gray')
   # ax2 = fig.add_subplot(122)
   # ax2.imshow(reduced_entropy, cmap='gray')
   # plt.show()

def entropy_diff(file1, file2):
   image1 = cv2.imread(file1, 0)
   image2 = cv2.imread(file2, 0)
   diff = bv.image_diff(image1, image2)

   # e1 = bv.entropy_image(image1)
   # e2 = bv.entropy_image(image2)
   # ediff = bv.image_diff(e1, e2)

   bui.show_images([image1, image2, diff])   

# run a test battery on all files in BSD test until user quits
def test_battery(f):
   folder = 'BSDS300/images/train'
   files = [f for f in listdir(folder) if isfile(join(folder, f))]
   for file in files:
      file = folder+'/'+file
      print(file)
      f(file)

# NOTE: does not seem useful
# display symmetry using tile by tile symmetry along row/column axes
def symmetry_tile_test(file):
   image = cv2.imread(file, 0)
   vsym = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   hsym = np.copy(vsym)
   tsym = np.copy(vsym)

   size = 50
   smap = bv.symmetry_tile_map(image, size)
   vmap = bv.symmetry_tile_vertical_map(image, 1, size)
   hmap = bv.symmetry_tile_horizontal_map(image, 0, size)
   # smap = bv.symmetry_strip_map(image, size)
   # vmap = bv.symmetry_axis_map(image, 1, size)
   # hmap = bv.symmetry_axis_map(image, 0, size)

   bui.paint_symmetry(tsym, smap, size)
   bui.paint_symmetry(vsym, vmap, size)
   bui.paint_symmetry(hsym, hmap, size)

   bui.show_images([image, vsym, hsym, tsym])

# NOTE: does not seem useful
# display symmetry using first approach, full width/height combined
# symmetry.
def symmetry_strip_test(file):
   image = cv2.imread(file, 0)
   vsym = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
   hsym = np.copy(vsym)
   tsym = np.copy(vsym)

   size = 10
   smap = bv.symmetry_strip_map(image, size)
   vmap = bv.symmetry_strip_axis_map(image, 1, size)
   hmap = bv.symmetry_strip_axis_map(image, 0, size)

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


# sobel derivative filter
def sobel_filter_test(file):
   image = cv2.imread(file, 0)
   blur = cv2.GaussianBlur(image,(3,3),0)
   laplacian = cv2.Laplacian(image, cv2.CV_64F, 3, 1)
   sobel = bv.sobel_filter(image, 3)
   entropy = bv.entropy_image(image, 7)
   height, width = image.shape[:2]

   # dpi = 80.0
   # xpixels, ypixels = 800, 800

   # fig = plt.figure(figsize=(height/dpi, width/dpi), dpi=dpi)
   # fig.figimage(image)
   # plt.show()

   # plt.subplot(1,3,1),plt.imshow(image,cmap = 'gray')
   # plt.title('Original'), plt.xticks([]), plt.yticks([])
   # plt.subplot(1,3,2),plt.imshow(laplacian,cmap = 'gray')
   # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
   # plt.subplot(1,3,3),plt.imshow(sobel,cmap = 'gray')
   # plt.title('Sobel'), plt.xticks([]), plt.yticks([])
   # plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
   # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
   # plt.show()
   bui.show_images([image, laplacian, sobel, entropy])

# display entropy of image components
def partial_entropy_test(file):
   image = cv2.imread(file, 0)
   height, width = image.shape[:2]
   colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

   ps = 3
   sizeY = int(height / ps)
   sizeX = int(width / ps)

   fsize = 50
   step = 10
   fsize = 10
   step = 5
   rsize = step

   images = [image]
   for y in range(0, ps):
      for x in range(0, ps):
         y1 = y*sizeY
         y2 = y1+sizeY
         x1 = x*sizeX
         x2 = x1 + sizeX

         nimage = image[y1:y2, x1:x2]
         emap = bv.entropy_map(nimage, fsize, step)
         cimage = cv2.cvtColor(nimage, cv2.COLOR_GRAY2RGB)
         bui.paint_map(cimage, emap, fsize, step, rsize)

         images.append(nimage)
         images.append(cimage)

   bui.show_images(images)

# display entropy calculations of various focus sizes side by side
def entropy_focus_test(file):
   image = cv2.imread(file, 0)
   image = cv2.GaussianBlur(image,(9,9),0)
   bui.show_images([image])

   images = [image]
   for f in [[80, 20], [50, 10], [10, 5], [5, 2]]:
      fsize = f[0]
      step = f[1]
      rsize = step

      emap = bv.entropy_map(image, fsize, step)
      cimage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
      bui.paint_map(cimage, emap, fsize, step, rsize)
      images.append(cimage)

   bui.show_images(images)

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
   image = cv2.GaussianBlur(image,(3,3),0)

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
   fsize = 5
   step = 2
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
   bui.show_images([image, gray, grayMap, reds, greens, blues, redMap, greenMap, blueMap])

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

# file = 'entropy_images/bwv.jpg'
# file = 'images/symmetrytest.jpg'
# file = 'images/bgrtest2.jpg' # color test
# file = 'BSDS300/images/train/87065.jpg' # lizard
# file = 'BSDS300/images/train/100075.jpg' # bears
# file = 'BSDS300/images/train/134052.jpg' # leopard
# file = "circletest.png"
# file = 'entropygirl.png'
# file = 'leopard-ecrop.jpg' # leopard
file = 'BSDS300/images/train/181018.jpg' # some girl
# file = 'BSDS300/images/train/15004.jpg' # lady in the market

# entropy_image("star.jpg")
# entropy_diff("circle1.jpg", "circle2.jpg")

entropy_image(file)

# sobel_filter_test("star.jpg")
# color_channel_entropy_test(file)

#image = cv2.imread(file, cv2.IMREAD_COLOR)
#bui.plot_images([image, image, image, image, image])
# fig = plt.figure()
# ax = fig.add_subplot(2, 3, 1)
# ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax2 = fig.add_subplot(2, 3, 2)
# ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

# test_battery(sobel_filter_test)
# test_battery(entropy_focus_test)
# partial_entropy_test(file)
# test_battery(symmetry_tile_test)
# symmetry_tile_test(file)
# test_battery(color_channel_entropy_test)
# colored_entropy_test(file)
# color_extraction_test(file)
# line_finder(file)
# super_map_test(file)
# hybrid_map_test(file)
# contrast_map_test(file)
# entropy_map_test(file)
# cross_entropy_test_battery()



cv2.destroyAllWindows()