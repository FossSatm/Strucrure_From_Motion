import os
import cv2 as cv

SIFT = "SIFT"
SURF = "SURF"
ORB = "ORB"
AKAZE = "AKAZE"


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

    def CAMERA_MATRIX(self):
        return self.camera.CAMERA_MATRIX()

    # ---------- #
    # SET IMAGES #
    # ---------- #

    def open_image(self, src: str):
        self.src = os.path.normpath(src)
        self.dir_src = os.path.dirname(self.src)
        dir_name = os.path.normpath(self.dir)
        dir_name = dir_name.split(os.sep)
        self.dir_name = dir_name[len(dir_name) - 1]
        basename = os.path.splitext(os.path.basename(self.src))
        self.name = basename[0]
        self.suffix = basename[1]
