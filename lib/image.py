import os
import cv2 as cv

FM_SIFT = "SIFT"
FM_SURF = "SURF"
FM_ORB = "ORB"
FM_AKAZE = "AKAZE"


class Image:
    def __init__(self):
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

        self.camera: Camera = Camera()

    # ---------------- #
    # RETURN FUNCTIONS #
    # ---------------- #

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

    # ---------- #
    # SET IMAGES #
    # ---------- #

    def img_open_image(self, src: str):
        self.src = os.path.normpath(src)
        self.dir_src = os.path.dirname(self.src)
        dir_name = os.path.normpath(self.dir)
        dir_name = dir_name.split(os.sep)
        self.dir_name = dir_name[len(dir_name) - 1]
        basename = os.path.splitext(os.path.basename(self.src))
        self.img_name = basename[0]
        self.img_suffix = basename[1]

    def img_find_features(self, flag=FM_AKAZE):
        """

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

        img = cv2.imread(self.src, cv2.IMREAD_GRAYSCALE)  # open image as grayscale
        kp, descr = method.detectAndCompute(img, None)  # detect and compute keypoints
        self.kp = kp  # set kp to image class
        self.descr = descr  # set descr to image class
        self.kp_size = len(self.kp)  # take the size of kp
        for i in range(0, self.kp_size):  # for all kps
            self.kp_ids.append(i)  # set an id the index of kp in list
