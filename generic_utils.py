import numpy as np

def intersect_list(list1, list2):
    return list(set(list1).intersection(set(list2)))

def difference_list(list1, list2):
    return list(set(list1)- set(list2))

def union_list(list1, list2):
    return list(set(list1).union(set(list2)))

def remove_nan_union(array_a: np.ndarray, array_b: np.ndarray):
    isnan_mask = np.logical_or(np.isnan(array_a), np.isnan(array_b))
    return np.delete(array_a, isnan_mask), np.delete(array_b, isnan_mask)