from __future__ import print_function
from __future__ import division

def quickSort(arr):
    """
    The QuickSort algorithm is an 
    efficient sorting algorithm,
    serving as a systematic method for 
    placing the elements of an array 
    in order.
    
    Args:
      :param arr: a list containing the 
                  elements to sort
      :return:  the sorted list of arguments
    """
    less = []
    pivotList = []
    more = []
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        for i in arr:
            if i < pivot:
                less.append(i)
            elif i > pivot:
                more.append(i)
            else:
                pivotList.append(i)
        less = quickSort(less)
        more = quickSort(more)
        return less + pivotList + more

if __name__ == "__main__":
    a = [4, 65, 2, -31, 0, 99, 83, 782, 1]
    a = quickSort(a)
    print(a)

