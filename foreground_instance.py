import numpy as np
from utils import get_overlap

class ForegroundInstance():
    def __init__(self, pnts_with_label, scene_id=None):
        self.scene_id = scene_id
        self.pnts_with_label = pnts_with_label 
        pnts = pnts_with_label[:, :3]
        self.center = np.mean(pnts, axis=1)
        self.corners = np.array([
            [pnts[:, 0].min(), pnts[:, 1].min(), pnts[:, 2].min()],
            [pnts[:, 0].max(), pnts[:, 1].max(), pnts[:, 2].max()]
        ])

    def get_center(self):
        return self.center 

    def get_point_cloud(self):
        return np.insert(self.pnts_with_label, 6, self.scene_id, axis=1)
        
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
             


