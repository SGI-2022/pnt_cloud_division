import numpy as np
from utils import get_overlap, compute_distance
from waymo_data import SegType

"""
A class that represents a subset of point cloud data as an instance.
"""
class Instance():
    """
    - pnts_with_label: (N, 5) numpy array where the colums are 
        (x, y, z, instance id, class label)
    - scene_id: a unique scene id of the point cloud that this instance belongs to
    """
    def __init__(self, pnts_with_label, scene_id=None):
        self.scene_id = scene_id
        self.pnts_with_label = pnts_with_label 
        pnts = pnts_with_label[:, :3]
        self.center = np.mean(pnts, axis=1)
        self.corners = np.array([
            [pnts[:, 0].min(), pnts[:, 1].min(), pnts[:, 2].min()],
            [pnts[:, 0].max(), pnts[:, 1].max(), pnts[:, 2].max()]
        ])
        self.instance_id = pnts_with_label[0, 3]
        self.class_label = SegType(pnts_with_label[0, 4])


    def get_center(self):
        return self.center 


    def get_point_cloud(self, with_scene_id=True):
        if with_scene_id:
            return np.insert(self.pnts_with_label, 6, self.scene_id, axis=1)
        else:
            return self.pnts_with_label
        
    """
    Check whether self is colliding with other.
    Return True if self and other belongs to the same scene, 
        or the overlapping volume of two bounding box is smaller than thresh*min_volume_of_two.
    """
    def is_colliding(self, other, thresh=0.1):
        if self.scene_id == other.scene_id:
            return False
        # Get corners of two instances
        [p1_min, p1_max] = self.corners 
        [p2_min, p2_max] = other.corners 
        # Compute volume of the bounding boxes
        box1_vol = np.prod(p1_max - p1_min)
        box2_vol = np.prod(p2_max - p2_min)
        # Compute volume of overlaping area 
        overlap = get_overlap(p1_min[0], p1_max[0], p2_min[0], p2_max[0]) \
                * get_overlap(p1_min[1], p1_max[1], p2_min[1], p2_max[1]) \
                * get_overlap(p1_min[2], p1_max[2], p2_min[2], p2_max[2])
        return overlap / min(box1_vol, box2_vol) > thresh


    """
    Subdivides a instance into a list of instances.
    """
    def subdivision(self, threshold=2.0):
        # # compute distance matrix 
        # dists = distance_matrix(self.pnts_with_label[:, :3], self.pnts_with_label[:, :3])

        # cluster points by cutting long edges 
        visited = np.full(len(self.pnts_with_label), False)
        objs = []
        while not np.all(visited):
            root = np.argwhere(visited == False)[0, 0]
            visited[root] = True
            cluster = [root]
            queue = [root]
            while len(queue) > 0:
                current_node = queue.pop(0)
                dists = compute_distance(self.pnts_with_label[current_node, :3], self.pnts_with_label[:, :3])
                new_nodes = np.argwhere((dists < threshold) & (visited == False)).flatten()
                if len(new_nodes) == 0:
                    continue
                visited[new_nodes] = True 
                cluster += list(new_nodes)
                queue += list(new_nodes)
            if len(cluster) < 2:
                continue
            objs.append(Instance(self.pnts_with_label[cluster], self.scene_id))
        return objs
            



