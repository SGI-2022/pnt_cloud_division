import numpy as np 
import polyscope as ps

###################### Visualization ######################

"""
Visualize point cloud with single color.

- pnts: (N, 3) numpy array of point coordinates
"""
def visualize_point_cloud(pnts, radius=3e-4):
    ps.init() 
    ps.set_up_dir('z_up')
    ps.register_point_cloud("my points", pnts, radius=radius)
    ps.show()


"""
Visualize point cloud with given color pallete. 
Points with no color assignment will be white.

- pnts: (N, 3) numpy array of point coordinates
- color_idx: (N, 1) numpy array of indices into color pallete
- colors: (k, 3) list of RGB values in range [0, 1]
"""
def visualize_point_cloud_colored(pnts, color_idx, colors, radius=8e-4):
    ps.init() 
    ps.set_up_dir('z_up')
    ps_cloud = ps.register_point_cloud("my points", pnts, radius=radius)
    pnts_color = np.ones_like(pnts)
    color_mask = (color_idx > -1)
    pnts_color[color_mask] = colors[color_idx[color_mask]]
    ps_cloud.add_color_quantity('grid_labels', pnts_color, enabled=True)
    ps.show()



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


###################### Main Script ######################

if __name__ == "__main__":
	pnts = np.load("WaymoExamples/0025.npy")[:, :3]

	grid_dim = (5, 5, 1)
	num_grids = grid_dim[0] * grid_dim[1] * grid_dim[2]
	grid_indices = split_to_grids(pnts, grid_dim)

	colors = np.random.random((num_grids, 3)) # generate a random color pallete
	visualize_point_cloud_colored(pnts, grid_indices, colors)