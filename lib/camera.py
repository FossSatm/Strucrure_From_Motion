import numpy as np


class Camera:
    def __init__(self):
        self.fx: float = 1.0
        self.fy: float = 1.0
        self.cx: float = 0.0
        self.cy: float = 0.0

    def CAMERA_MATRIX(self):
        """
        Creates and return the camera matrix.
        :return: mtrx
        """
        mtrx = [[fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]]
        mtrx = np.array(mtrx)
        return mtrx

    def set_camera_parameters(self, fx: float = 1.0, fy: float = 1.0, cx: float = 0.0, cy: float = 0.0):
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy
