
#####################################################################################
# Sorting
#####################################################################################

# many default sort algorithms do the following:
# -> check size of input
# -> if small: use insertion sort
# -> if large: use quicksort
# -> if poor pivots: use merge sort

# bubble sort - exchanges adjacent elements in multiple passes

# insertion sort O(n^2) - good for small inputs
def insertionSort(arr):
   for i in range(1, len(arr)):
      current = arr[i]
      j = i

      while j > 0 and arr[j-1] > current:
         arr[j] = arr[j-1]
         j--

      arr[j] = current

print(insertionSort([54,26,93,17,77,31,44,55,20]))

# shell sort - improves on insertion sort by using sublists

# quick sort
def quickSort(arr):

# merge sort

# heap sort

#####################################################################################
# Trees
#####################################################################################

# AVL tree