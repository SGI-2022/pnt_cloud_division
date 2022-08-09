# pnt_cloud_division
## Installation
Dependencies: numpy, polyscope

I used Annaconda to manage the environment, so simply run 

`conda install numpy`

`conda install -c conda-forge polyscope`

## Run
`point_cloud_div.py` contains functions for dividing a point cloud into grids/instances/foreground and visualization. 
The data is not included in the repo.
You would need to download the `WaymoExamples` data from the shared google drive, and put it 
under the root folder of this project. Alternatively, you could edit the path in the main
script of `point_cloud_div.py` in order to correctly load the point cloud data.

Then you can run `python point_cloud_div.py` to see the visualization.
