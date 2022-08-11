from enum import Enum

"""
Data Source: 
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/segmentation.proto
"""
class SegType(Enum):
    UNDEFINED = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
    # vehicles, RV, limo, tram).
    OTHER_VEHICLE = 4
    MOTORCYCLIST = 5
    BICYCLIST = 6
    PEDESTRIAN = 7
    SIGN = 8
    TRAFFIC_LIGHT = 9
    # Lamp post, traffic sign pole etc.
    POLE = 10
    # Construction cone/pole.
    CONSTRUCTION_CONE = 11
    BICYCLE = 12
    MOTORCYCLE = 13
    BUILDING = 14
    # Bushes, tree branches, tall grasses, flowers etc.
    VEGETATION = 15
    TREE_TRUNK = 16
    # Curb on the edge of roads. This does not include road boundaries if
    # there’s no curb.
    CURB = 17
    # Surface a vehicle could drive on. This include the driveway connecting
    # parking lot and road over a section of sidewalk.
    ROAD = 18
    # Marking on the road that’s specifically for defining lanes such as
    # single/double white/yellow lines.
    LANE_MARKER = 19
    # Marking on the road other than lane markers, bumps, cateyes, railtracks
    # etc.
    OTHER_GROUND = 20
    # Most horizontal surface that’s not drivable, e.g. grassy hill,
    # pedestrian walkway stairs etc.
    WALKABLE = 21
    # Nicely paved walkable surface when pedestrians most likely to walk on.
    SIDEWALK = 22


"""
Check whether a given label or array of labels is foreground object,
i.e. label 1 to 13.

- label: int or (N, 1) numpy array of class label
"""
def is_foreground(label):
    if isinstance(label, int):
        return label > SegType.UNDEFINED.value and label < SegType.BUILDING.value
    else:
        return (label > SegType.UNDEFINED.value) & (label < SegType.BUILDING.value)


"""
Check whether a given label or array of labels is vegetation (15)

- label: int or (N, 1) numpy array of class label
"""
def is_vegetation(label):
    if isinstance(label, int):
        return label == SegType.VEGETATION.value 
    else:
        return label == SegType.VEGETATION.value