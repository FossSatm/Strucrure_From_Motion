# Python Libraries #
import os

# OpenCV #
import cv2 as cv

# Written Libraries #
from lib.camera import *

FM_SIFT = "SIFT"
FM_SURF = "SURF"
FM_ORB = "ORB"
FM_AKAZE = "AKAZE"

CAM_DEFAULT = 0
CAM_APPROXIMATE = 1
CAM_FROM_FILE = 2


class Image:
    def __init__(self):
        self.img_id: int = 0
        self.src: str = ""
        self.dir_src: str = ""
        self.dir_name: str = ""
        self.img_name: str = ""
        self.img_suffix: str = ""

        self.width: int = 0
        self.height: int = 0
        self.channels: int = 0

        self.kp: [] = []
        self.descr: [] = []
        self.kp_ids: [] = []
        self.kp_size: int = 0

        self.feature_list: [] = []
        self.feature_colors: [] = []
        self.feature_ids: [] = []

        self.camera: Camera = Camera()

    # ---------------- #
    # RETURN FUNCTIONS #
    # ---------------- #
    def IMG_ID(self):
        return self.img_id

    def SRC(self):
        return self.src

    def DIR_SRC(self):
        return self.dir_src

    def DIR_NAME(self):
        return self.dir_name

    def IMG_NAME(self):
        return self.img_name

    def IMG_SUFFIX(self):
        return self.img_suffix

    def WIDTH(self):
        return self.width

    def HEIGHT(self):
        return self.height

    def CHANNELS(self):
        return self.channels

    def KEYPOINT_LIST(self):
        return self.kp

    def KEYPOINT_AT(self, index: int):
        return self.kp[index]

    def DESCRIPTOR_LIST(self):
        return self.descr

    def DESCRIPTOR_AT(self, index: int):
        return self.descr[index]

    def KEYPOINTS_LIST_SIZE(self):
        return self.kp_size

    def KEYPOINT_IDS_LIST(self):
        return self.kp_ids

    def KEYPOINT_IDS_AT(self, index: int):
        return self.kp_ids[index]

    def CAMERA_MATRIX(self):
        return self.camera.CAMERA_MATRIX()

    def INFO(self):
        info = "\n"
        info += "Image Name: %s\n" % (self.img_name + self.img_suffix)
        info += "Image ID: %d\n" % self.img_id
        info += "Dimensions: %d" % self.width + " x %d" % self.height + " x %d\n" % self.channels
        info += "Focal Length fx: %.2f\n" % self.camera.FX()
        info += "             fy: %.2f\n" % self.camera.FY()
        info += "Features: %d\n" % len(self.kp)
        return info

    def INFO_DEBUGGING(self):
        info = "\n"
        info += "img_id: %d\n" % self.img_id
        info += "src: %s\n" % self.src
        info += "dir_src: %s\n" % self.dir_src
        info += "dir_name: %s\n" % self.dir_name
        info += "img_name: %s\n" % self.img_name
        info += "img_suffix: %s\n" % self.img_suffix
        info += "width: %d\n" % self.width
        info += "height: %d\n" % self.height
        info += "channels: %d\n" % self.channels
        info += "feature_points_size: %d\n" % len(self.kp)
        return info

    # ---------- #
    # SET IMAGES #
    # ---------- #

    def img_open_image(self, src: str, img_id=0):
        self.img_id = img_id
        self.src = os.path.normpath(src)
        self.dir_src = os.path.dirname(self.src)
        dir_name = os.path.normpath(self.dir_src)
        dir_name = dir_name.split(os.sep)
        self.dir_name = dir_name[len(dir_name) - 1]
        basename = os.path.splitext(os.path.basename(self.src))
        self.img_name = basename[0]
        self.img_suffix = basename[1]

    def img_set_camera(self, flag=CAM_DEFAULT):
        if flag == CAM_APPROXIMATE:
            self.camera.camera_approximate_parameters(self.width, self.height)
        elif flag == CAM_FROM_FILE:
            pass
        else:
            self.set_camera_parameters()

    def img_find_features(self, flag=FM_AKAZE, set_camera_method=CAM_DEFAULT):
        """
        :param set_camera_method:
        :param flag:
        :return:
        """
        # Choose the Feature Points Extraction method
        if flag == FM_SIFT:
            method = cv.xfeatures2d.SIFT_create(nfeatures=F_THRESHOLD, edgeThreshold=10, contrastThreshold=0.1)
        elif flag == FM_SURF:
            method = cv.xfeatures2d.SURF_create(hessianThreshold=3000, nOctaves=4, nOctaveLayers=2, upright=0)
        elif flag == FM_ORB:
            method = cv.ORB_create()
        else:
            method = cv.AKAZE_create()

        img = cv.imread(self.src)  # open image as grayscale
        img_size = img.shape  # take the shape of image (height x width x channels)
        self.width = img_size[1]  # set width
        self.height = img_size[0]  # set height
        self.channels = 1  # assume that the image is grayscale
        if len(img_size) == 3:  # check if image is really grayscale
            self.channels = img_size[2]  # if not grayscale take the size of channels

        self.img_set_camera(flag=set_camera_method)

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kp, descr = method.detectAndCompute(img, None)  # detect and compute keypoints
        self.kp = np.array(kp)  # set kp to image class
        self.descr = np.array(descr)  # set descr to image class
        self.kp_size = len(self.kp)  # take the size of kp
        for i in range(0, self.kp_size):  # for all kps
            self.kp_ids.append(i)  # set an id the index of kp in list
        self.kp_ids = np.array(self.kp_ids)

    def img_set_features(self, feat_list, color_list, feat_ids):
        self.feature_list = feat_list
        self.feature_colors = color_list
        self.feature_ids = feat_ids
