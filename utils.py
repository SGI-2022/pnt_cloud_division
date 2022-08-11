import numpy as np

"""
Return the length of overlapping area between two line segments 
[v1_min, v1_max] and [v2_min, v2_max].
"""
def get_overlap(v1_min, v1_max, v2_min, v2_max):
    if v1_min > v2_min and v1_min < v2_max:
        return min(v1_max, v2_max) - v1_min 
    if v2_min > v1_min and v2_min < v1_max:
        return min(v1_max, v2_max) - v2_min 
    return 0


"""
Compute euclidean distances from p to pnts.

- p: (3) numpy array of a point
- pnts: (N, 3) numpy array of points
"""
def compute_distance(p, pnts):
    return np.sum((pnts - p) ** 2, axis=1)