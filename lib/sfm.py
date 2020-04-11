from lib.image import *
import os
import glob


class SFM:
    def __init__(self):
        self.ALL_SUPPORTED_FORMATS_LIST = ["*.jpg", "*.jpeg", "*.jpe", "*.png", "*.bmp",
                                           "*.tif", "*.tiff", "*.dib", "*.pbm", "*.pgm",
                                           "*.ppm", "*.sr", "*.ras"]
        self.image_list: [] = []

    def sfm_run(self, f_src: str):
        self.sfm_set_image_list(f_src)

    def sfm_set_image_list(self, f_src):
        for suffix in self.ALL_SUPPORTED_FORMATS_LIST:
            imgs_src = os.path.normpath(f_src) + os.path.normpath("/" + suffix)
            img_files_path = glob.glob(imgs_src)
            for path in img_files_path:
                img_tmp = Image()
                img_tmp.img_open_image(path)
                self.image_list.append(img_tmp)
                # print(img_tmp.IMG_NAME())

        # print(len(self.image_list))
        for img in self.image_list:
            print(img.IMG_NAME())
