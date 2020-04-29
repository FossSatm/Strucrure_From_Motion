# Strucrure_From_Motion
A structure from motion algorithm. Written in python.

## Dependences:
- **python3 64-bit:** Need this architecture to run in large images without memory error.
- **numpy:** pip install numpy
- **opencv:** pip install opencv-python
- **scikit-learn:** pip install scikit-learn
- **open3d:** pip install open3d-python open3d==0.9.0.0

**One line command:** pip install numpy opencv-python scikit-learn open3d-python open3d==0.9.0.0

## Simple Description:

Currently this algorithm can read a block of image in a given folder and create an export
folder with name **sfm_tmp** in which it stores the all the exported point clouds. Also it
uses **sklearn.cluster.dbscan** to remove the noise from the point cloud.
