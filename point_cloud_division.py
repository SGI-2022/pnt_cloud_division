import numpy as np 
from visualize import visualize_point_cloud_colored
from waymo_data import is_foreground
from foreground_instance import ForegroundInstance
from random import shuffle

###################### Grid Division ######################

"""
Helper function for `split_to_grids(pnts, grid_dim)`.

Compute the boundaries for each grid. 
Return a list of boundaries, where each element consists
	[[x_low_bound, x_up_bound], [y_low_bound, y_up_bound], [z_low_bound, z_up_bound]]
of each grid.

- pnts_bound: boundaries of the point cloud (see first line in this function for format)
- grid_dim: list of 3 integers defining the dimensions of the grids
"""
def get_grid_bounds(pnts_bound, grid_dim):
	[[pnts_x_min, pnts_x_max], [pnts_y_min, pnts_y_max], [pnts_z_min, pnts_z_max]] = pnts_bound 
	x_side_length = (pnts_x_max - pnts_x_min) / grid_dim[0]
	y_side_length = (pnts_y_max - pnts_y_min) / grid_dim[1]
	z_side_length = (pnts_z_max - pnts_z_min) / grid_dim[2] 
	grid_bounds = []
	
	x_low = pnts_x_min
	for _ in range(grid_dim[0]):
		x_up = x_low + x_side_length
		y_low = pnts_y_min
		for _ in range(grid_dim[1]):
			y_up = y_low + y_side_length
			z_low = pnts_z_min
			for _ in range(grid_dim[2]):
				z_up = z_low + z_side_length
				grid_bounds.append([
					[x_low, x_up],
					[y_low, y_up],
					[z_low, z_up],
				])
				z_low = z_up 
			y_low = y_up
		x_low = x_up 
	return grid_bounds 


"""
Helper function for `split_to_grids(pnts, grid_dim)`.

Return a (N, 1) numpy array of grid indices, where a grid index of -1 means no assignment.

- pnts: (N, 3) numpy array of point coordinates
- grid_bounds: list of boundaries for each grid, with format
	[[x_low_bound, x_up_bound], [y_low_bound, y_up_bound], [z_low_bound, z_up_bound]]
  for each element.
"""
def add_grid_index(pnts, grid_bounds):
	grid_indices = np.full(len(pnts), -1).astype(int)
	for (i, bound) in enumerate(grid_bounds):
		[[x_min, x_max], [y_min, y_max], [z_min, z_max]] = bound 
		mask = (pnts[:, 0] >= x_min) & (pnts[:, 0] <= x_max) \
		  	 & (pnts[:, 1] >= y_min) & (pnts[:, 1] <= y_max) \
			 & (pnts[:, 2] >= z_min) & (pnts[:, 2] <= z_max) 
		grid_indices[mask] = i 
	return grid_indices


"""
Split the point cloud into grids of given dimension. 
Return a (N, 1) numpy array of grid indices, where a grid index of -1 means no assignment.

- pnts: (N, 3) numpy array of point coordinates
- grid_dim: list of 3 integers defining the dimensions of the grids
"""
def split_to_grids(pnts, grid_dim):
	pnts_bound = [
		[pnts[:, 0].min(), pnts[:, 0].max()],
		[pnts[:, 1].min(), pnts[:, 1].max()],
		[pnts[:, 2].min(), pnts[:, 2].max()]
	]
	grid_bounds = get_grid_bounds(pnts_bound, grid_dim)
	grid_indices = add_grid_index(pnts, grid_bounds)
	return grid_indices


###################### Label Division ######################

"""
Split the point cloud into foreground (label 1) and background (label 0).
Return a (N, 1) numpy array of label indices.

- seg_labels: (N, 1) numpy array of class labels
"""
def split_foreground_background(seg_labels):
	return is_foreground(seg_labels).astype(int)


"""
Split the point cloud by unique (instance id, class label) pairs.
Return a (N, 1) numpy array of unique instance indices.

- labels: (N, 2) numpy array of (instance id, class label) pairs.
"""
def split_unique_instance(labels):
	unique_instance_ids = np.full(len(labels), -1).astype(int)
	instance_labels = np.unique(labels, axis=0)
	for i in range(len(instance_labels)):
		unique_label = instance_labels[i]
		selection_mask = np.all(labels == unique_label, axis=1)
		unique_instance_ids[selection_mask] = i 
	return unique_instance_ids


###################### Object Centric Division ######################
def cluster_instances(pnts_with_label, scene_id):
	objects = []
	instance_labels = pnts_with_label[:, 5]
	for label in np.unique(instance_labels):
		objects.append(
			ForegroundInstance(pnts_with_label[instance_labels==label], scene_id)
		)
	return objects

def object_centric_assemble(pnts_with_label1, pnts_with_label2, collision_thresh=0.1):
	# Cluster the point cloud by instances
	objects = cluster_instances(pnts_with_label1, 0) + cluster_instances(pnts_with_label2, 1)
	# Randomize the order of objects
	shuffle(objects)

	# Select a subset of non-overlapping objects
	selected_objects = []
	for obj in objects:
		collided = False
		for selected_obj in selected_objects:
			if obj.is_colliding(selected_obj, collision_thresh):
				collided = True
				break 
		if not collided:
			selected_objects.append(obj)

	# Take the union of the selected objects
	new_pnts = np.concatenate([obj.get_point_cloud() for obj in selected_objects], axis=0)
	return new_pnts
	


###################### Main Script ######################

def grid_demo():
	pnts = np.load("WaymoExamples/0025.npy")[:, :3]

	grid_dim = (5, 5, 1)
	num_grids = grid_dim[0] * grid_dim[1] * grid_dim[2]
	grid_indices = split_to_grids(pnts, grid_dim)

	colors = np.random.random((num_grids, 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, grid_indices, colors)


def foreground_demo():
	seg_labels = np.load("WaymoExamples/0025_seg.npy")[:, 1]
	pnts = np.load("WaymoExamples/0025.npy")[:len(seg_labels), :3]

	split_indices = split_foreground_background(seg_labels)

	colors = np.random.random((2, 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, split_indices, colors)

def instance_demo():
	labels = np.load("WaymoExamples/0025_seg.npy")
	pnts = np.load("WaymoExamples/0025.npy")[:len(labels), :3]

	indices = split_unique_instance(labels)

	colors = np.random.random((len(indices), 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, indices, colors)

def _load_data(id):
	labels = np.load("WaymoExamples/"+id+"_seg.npy")
	pnts = np.load("WaymoExamples/"+id+".npy")[:len(labels), :3]

	fore_back_split = split_foreground_background(labels[:, 1])
	pnts_back = pnts[fore_back_split==0]
	pnts_fore = pnts[fore_back_split==1]
	labels_fore = labels[fore_back_split==1]

	instance_split = split_unique_instance(labels_fore)
	instance_split = instance_split.reshape((len(instance_split), 1))

	pnts_fore_with_label = np.hstack([pnts_fore, labels_fore, instance_split])
	return pnts_back, pnts_fore_with_label

def object_centric_demo():
	pnts_back1, pnts_fore_with_label1 = _load_data("0025")
	_, pnts_fore_with_label2 = _load_data("0084")
	pnts_fore_with_label_merged = object_centric_assemble(pnts_fore_with_label1, pnts_fore_with_label2)
	new_pnts = np.concatenate([pnts_back1, pnts_fore_with_label_merged[:, :3]], axis=0)
	color_labels = np.concatenate([
		np.full(len(pnts_back1), 2),
		pnts_fore_with_label_merged[:, 6]
	], axis=0).astype(int)
	colors = np.array([
		[0.6, 0.2, 0.3],
		[0.1, 0.3, 0.6],
		[0.8, 0.4, 0.4]
	])
	visualize_point_cloud_colored(new_pnts, color_labels, colors)


def union_demo():
	pnts_back1, pnts_fore_with_label1 = _load_data("0025")
	_, pnts_fore_with_label2 = _load_data("0084")

	new_pnts = np.concatenate([pnts_back1, pnts_fore_with_label1[:, :3], pnts_fore_with_label2[:, :3]], axis=0)
	color_labels = np.concatenate([
		np.full(len(pnts_back1), 2),
		np.full(len(pnts_fore_with_label1), 0),
		np.full(len(pnts_fore_with_label2), 1)
	], axis=0).astype(int)
	colors = np.array([
		[0.6, 0.2, 0.3],
		[0.1, 0.3, 0.6],
		[0.8, 0.4, 0.4]
	])
	visualize_point_cloud_colored(new_pnts, color_labels, colors)

if __name__ == "__main__":
	# grid_demo()
	# foreground_demo()
	# instance_demo()
	object_centric_demo()
	# union_demo()