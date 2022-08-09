import numpy as np 
import polyscope as ps

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