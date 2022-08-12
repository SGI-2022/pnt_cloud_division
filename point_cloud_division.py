import numpy as np 
from visualize import visualize_point_cloud_colored
from waymo_data import is_foreground, is_vegetation
from instance import Instance
from random import shuffle
from utils import compute_distance


"""
Divide the input point cloud data into subdivision. Detailed documentation
are attached to the functions.

Inputs
- pnts: (N, 3) point cloud coordinates
- labels: (N, 2) instance id and class label pairs, see `waymo_data.py` for more semantic meanings
- ... additional parameters, see below

Output
- div_ids: (N, 1) labels of subdivision for each point
"""
################################################################
###################### Division Functions ######################
################################################################

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
	div_ids = add_grid_index(pnts, grid_bounds)
	return div_ids


"""
Split a list of point clouds into grids of given dimension. 
This function ensures that all point clouds are cut in the same way.
Return a list of (N, 1) numpy arrays of grid indices, where a grid index of -1 means no assignment.

- pnts: list of (N, 3) numpy arrays of point coordinates
- grid_dim: list of 3 integers defining the dimensions of the grids
- ref_pc_idx: index of the point cloud used to create grid frames, default to 0
"""
def split_to_grids_multiple(pnts_lst, grid_dim, ref_pc_idx=0):
	ref_pnts = pnts_lst[ref_pc_idx]
	pnts_bound = [
		[ref_pnts[:, 0].min(), ref_pnts[:, 0].max()],
		[ref_pnts[:, 1].min(), ref_pnts[:, 1].max()],
		[ref_pnts[:, 2].min(), ref_pnts[:, 2].max()]
	]
	grid_bounds = get_grid_bounds(pnts_bound, grid_dim)
	div_ids_lst = [add_grid_index(pnts, grid_bounds) for pnts in pnts_lst]
	return div_ids_lst


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


"""
Split the point cloud into foreground (label 1) and background (label 0).
Return a (N, 1) numpy array of label indices.

- seg_labels: (N, 1) numpy array of class labels
"""
def split_foreground_background(seg_labels):
	return is_foreground(seg_labels).astype(int)


##############################################################
###################### Helper Functions ######################
##############################################################

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


###################### Object Centric Division ######################
def cluster_instances(pnts, scene_id):
	objects = []
	instance_labels = split_unique_instance(pnts[:, 3:])
	for label in np.unique(instance_labels):
		objects.append(
			Instance(pnts[instance_labels==label], scene_id)
		)
	return objects


def join_instances(objs1, objs2, collision_thresh=0.1):
	objects = objs1 + objs2
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

	return selected_objects


def object_centric_assemble(pnts_with_label1, pnts_with_label2, collision_thresh=0.1):
	# Cluster the point cloud by instances
	objs1 = cluster_instances(pnts_with_label1, 0) 
	objs2 = cluster_instances(pnts_with_label2, 1)

	# Select a subset of non-overlapping objects
	selected_objects = join_instances(objs1, objs2, collision_thresh)

	# Take the union of the selected objects
	new_pnts = np.concatenate([obj.get_point_cloud() for obj in selected_objects], axis=0)
	color_labels = np.concatenate([obj.get_scene_id(broadcast=True) for obj in selected_objects], axis=0)
	return new_pnts, color_labels


def object_centric_assemble_with_background(
	pnts_with_label1, 
	pnts_with_label2, 
	pnts_back2, 
	background_range=1.5,
	collision_thresh=0.1):
	# Cluster the point cloud by instances
	objs1 = cluster_instances(pnts_with_label1, 0) 
	objs2 = cluster_instances(pnts_with_label2, 1)
	
	# Select a subset of non-overlapping objects
	selected_objects = join_instances(objs1, objs2, collision_thresh)
	
	# Include some background of scene 2
	selected_back = []
	for obj in selected_objects:
		if obj.get_scene_id() == 0:
			continue 
		# center = obj.get_center()
		center = obj.get_box_center()
		radius = obj.get_radius() * background_range
		
		background_dists = compute_distance(center, pnts_back2[:, :3])
		selected_back.append(pnts_back2[background_dists<radius])
	
	selected_back = np.concatenate(selected_back, axis=0)

	# Take the union of the selected objects
	new_pnts = np.concatenate([obj.get_point_cloud() for obj in selected_objects], axis=0)
	# Add the background points from scene 2
	new_pnts = np.concatenate([new_pnts, selected_back], axis=0)

	# Generate color labels 
	color_labels = np.concatenate([obj.get_scene_id(broadcast=True) for obj in selected_objects], axis=0)
	color_labels = np.concatenate([color_labels, np.full(len(selected_back), 3)], axis=0)

	return new_pnts, color_labels
	


"""
Choose a demo function to run as main script and see the visualization. 
"""
##########################################################
###################### Demo Scripts ######################
##########################################################

"""
Divide a point cloud into grids.
"""
def grid_demo():
	pnts = np.load("WaymoExamples/0025.npy")[:, :3]

	grid_dim = (5, 5, 1)
	num_grids = grid_dim[0] * grid_dim[1] * grid_dim[2]
	grid_indices = split_to_grids(pnts, grid_dim)

	colors = np.random.random((num_grids, 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, grid_indices, colors)


"""
Separate a point cloud by foreground/background.
"""
def foreground_demo():
	seg_labels = np.load("WaymoExamples/0025_seg.npy")[:, 1]
	pnts = np.load("WaymoExamples/0025.npy")[:len(seg_labels), :3]

	split_indices = split_foreground_background(seg_labels)

	colors = np.random.random((2, 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, split_indices, colors)


"""
Separate a point cloud by unique instance.
"""
def instance_demo():
	labels = np.load("WaymoExamples/0025_seg.npy")
	pnts = np.load("WaymoExamples/0025.npy")[:len(labels), :3]

	indices = split_unique_instance(labels)

	colors = np.random.random((len(indices), 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, indices, colors)


def _load_data(id):
	labels = np.load("WaymoExamples/"+id+"_seg.npy")
	pnts = np.load("WaymoExamples/"+id+".npy")[:len(labels), :3]
	pnts_with_labels = np.hstack([pnts, labels])

	fore_back_split = split_foreground_background(labels[:, 1])
	pnts_fore = pnts_with_labels[fore_back_split==1]
	pnts_back = pnts_with_labels[fore_back_split==0]
	
	return pnts_fore, pnts_back


"""
Merge two scenes by object centric method.
Use background of 0025 and a non-overlapping combination of the foreground objects 
in both 0025 and 0084.
"""
def object_centric_demo(blend_background=False):
	pnts_fore1, pnts_back1 = _load_data("0025")
	pnts_fore2, pnts_back2 = _load_data("0084")
	
	if blend_background:
		pnts_merged, color_labels = object_centric_assemble_with_background(pnts_fore1, pnts_fore2, pnts_back2, 3)
	else:
		pnts_merged, color_labels = object_centric_assemble(pnts_fore1, pnts_fore2)
	new_pnts = np.concatenate([pnts_back1, pnts_merged], axis=0)[:, :3]

	color_labels = np.concatenate([
		np.full(len(pnts_back1), 2),
		color_labels
	], axis=0).astype(int)

	colors = np.array([
		[31, 70, 144],
		[255, 165, 0],
		[108, 141, 210],
		[255, 229, 180]
	]) / 255

	visualize_point_cloud_colored(new_pnts, color_labels, colors)


"""
Merge two scenes directly (as a comparison to object centric)
Use background of 0025 and a union of the foreground objects in both 0025 and 0084.
"""
def union_demo():
	pnts_fore1, pnts_back1 = _load_data("0025")
	pnts_fore2, pnts_back2 = _load_data("0084")

	new_pnts = np.concatenate([pnts_back1, pnts_fore1, pnts_fore2], axis=0)[:, :3]
	color_labels = np.concatenate([
		np.full(len(pnts_back1), 2),
		np.full(len(pnts_fore1), 0),
		np.full(len(pnts_fore2), 1)
	], axis=0).astype(int)
	colors = np.array([
		[31, 70, 144],
		[255, 165, 0],
		[108, 141, 210],
		[255, 229, 180]
	]) / 255
	visualize_point_cloud_colored(new_pnts, color_labels, colors)


"""
Segment the vegetation in 0025 as instances by clustering.
"""
def segmentation_demo():
	id = "0025"
	labels = np.load("WaymoExamples/"+id+"_seg.npy")
	pnts = np.load("WaymoExamples/"+id+".npy")[:len(labels), :3]

	pnts_labels = np.hstack([pnts, labels])[::5] # subsampled to save time

	vege_mask = is_vegetation(pnts_labels[:, 4])
	pnts_labels_vege = pnts_labels[vege_mask]
	pnts_labels_other = pnts_labels[np.invert(vege_mask)]

	vege_instance = Instance(pnts_labels_vege)
	vege_pieces = vege_instance.subdivision(3)

	new_pnts = np.insert(pnts_labels_other, 5, -1, axis=1)
	for (i, obj) in enumerate(vege_pieces):
		labeled_obj = np.insert(obj.get_point_cloud(), 5, i, axis=1)
		new_pnts = np.concatenate([new_pnts, labeled_obj], axis=0)

	colors = np.random.random((len(vege_pieces), 3))
	visualize_point_cloud_colored(new_pnts[:, :3], new_pnts[:, 5].astype(int), colors)

if __name__ == "__main__":
	# grid_demo()
	# foreground_demo()
	# instance_demo()
	object_centric_demo(True)
	# union_demo()
	# segmentation_demo()