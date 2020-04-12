# Python Libraries #
import numpy as np


class Camera:
    """
    This class describe the camera characteristics.
    """
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
        """
        Set the parameters of the camera. This function can be used without input to set the default parameters.
        :param fx: Focal length for the width
        :param fy: Focal length for the height
        :param cx: Centroid x coordinate
        :param cy: Centroid y coordinate
        :return: Nothing
        """
        self.fx = fx  # Set fx
        self.fy = fy  # Set fy
        self.cx = cx  # Set cx
        self.cy = cy  # Set cy

    def approximate_focal_length(self, width, height):
        """
        Approximate the focal length using the following statement:
            0.7w <= f <= w
        We assume the f is the half of (0.7w + w). This gives the highest possibility to be as close as we can to
        the real focal length.
        As w we take the bigger size of the image.
        :param width: The width of the image (in pixel)
        :param height: The height of the image (in pixel)
        :return: focal
        """
        if width > height:  # Check id width > height
            w = width  # Set w = width
        else:  # else
            w = height  # w = height
        focal = (0.7 * w + w) / 2  # Approximate the focal length as the the average of (70% of w + 100% of w)
        return focal

    def camera_approximate_parameters(self, width: int, height: int):
        """
        Approximate the camera parameters using the width or height of the image.
        :param width: The width of the image (in pixel)
        :param height: The height of the image (in pixel)
        :return: Nothing
        """
        focal = self.approximate_focal_length(width, height)  # Find the focal length
        self.fx = focal  # We assume the pixel is square and set the same focal as fy
        self.fy = focal  # We assume the pixel is square and set the same focal as fx
        self.cx = width / 2  # Take the half width as center_x coordinate
        self.cy = height / 2  # Take the half height as center_y coordinate
