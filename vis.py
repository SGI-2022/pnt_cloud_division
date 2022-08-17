import numpy as np
import open3d as o3d 
import os
import polyscope as ps
from natsort import natsorted


seg_class_color= [
            [0.3,0.3,0.3], # 0
            [1,0,0],
            [1,0,0],
            [0.6, 0.1, 0.8], # 3
            [0.2, 0.1, 0.9],
            [0.5, 1, 0.5],
            [0,1,0], # 6
            [0.8,0.8,0.8],
            [0.0, 0.8, 0.8],
            [0.05, 0.05, 0.3],
            [0.8, 0.6, 0.2], # 10 
            [0.5, 1, 0.5],
            [0.5, 1, 0.5], # 12
            [0.2, 0.5, 0.8],
            [0.0, 0.8, 0],
            [0.0, 0.0, 0.0],
            [1, 0.0, 0.0], # 16
            [0.8, 0.2, 0.8],
            [1, 0, 1],
            [1, 0, 1], # 18
            [0., 1, 0.3],
            [0.9, 0.35, 0.2],
            [0.9, 0.6, 0.2], # 21
          ]

seg_class_color = np.array(seg_class_color)

ps.init()
ps.set_up_dir('z_up')

path = "/home/sudarshan/Documents/SGI/data_augmentation/Dataset1/segment-17065833287841703_2980_000_3000_000_with_camera_labels" 

data_files =  natsorted( os.listdir(path ) )


for file in data_files:

	if file.endswith(".npy") and (not file.endswith("seg.npy") ) :

		FilePCD = os.path.join(path, file)

	if( file.endswith("seg.npy")):

		SegFile = os.path.join(path , file)

		seg_labels = np.load(SegFile)[:, 1]
		points = np.load(FilePCD)[:seg_labels.shape[0], :3]
		ps_p = ps.register_point_cloud('points', points, radius=2e-4)
		ps_p.add_color_quantity('seg_labels', seg_class_color[seg_labels])
		ps.show()
