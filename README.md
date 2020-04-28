# Strucrure_From_Motion
A structure from motion algorithm. Written in python.

## Dependences:
- **numpy:** pip install numpy
- **opencv:** pip install opencv-python
- **scikit-learn:** pip install scikit-learn

## Simple Description:

Currently this algorithm can read a block of image in a given folder and create an export
folder with name **sfm_tmp** in which it stores the all the exported point clouds. Also it
uses **sklearn.cluster.dbscan** to remove the noise from the point cloud.
