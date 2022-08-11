# pnt_cloud_division
## Installation
Dependencies: numpy, scipy, polyscope

I used Annaconda to manage the environment, so simply run 

`conda install numpy scipy`

`conda install -c conda-forge polyscope`

## Use
### Interface
See the section of `Division Functions` at the top of `point_cloud_division.py`.

### Run
Download the `WaymoExamples` data from the shared google drive, and put it 
under the root folder of this project. Alternatively, you could edit the path in the main
script of `point_cloud_division.py` in order to correctly load the point cloud data.

Then you can run `python point_cloud_division.py` to see the visualization.
