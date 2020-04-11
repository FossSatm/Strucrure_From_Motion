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
        mtrx = [[self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0]]
        mtrx = np.array(mtrx)
        return mtrx

    def FX(self):
        return self.fx

    def FY(self):
        return self.fy

    def CX(self):
        return self.cx

    def CY(self):
        return self.cy

    def set_camera_parameters(self, fx: float = 1.0, fy: float = 1.0, cx: float = 0.0, cy: float = 0.0):
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy

    def approximate_focal_length(self, width, height):
        if width > height:  # Check id width > height
            w = width  # Set w = width
        else:  # else
            w = height  # w = height
        focal = (0.7 * w + w) / 2  # Approximate the focal length as the the average of (70% of w + 100% of w)
        return focal

    def camera_approximate_parameters(self, width: int, height: int):
        focal = self.approximate_focal_length(width, height)
        self.fx = focal
        self.fy = focal
        self.cx = width / 2
        self.cy = height / 2
