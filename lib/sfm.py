from lib.image import *
import os
import glob


class SFM:
    def __init__(self):
        # Create a list with all OpenCV supported image file formats
        self.ALL_SUPPORTED_FORMATS_LIST = ["*.jpg", "*.jpeg", "*.jpe", "*.png", "*.bmp",
                                           "*.tif", "*.tiff", "*.dib", "*.pbm", "*.pgm",
                                           "*.ppm", "*.sr", "*.ras"]
        self.image_list: [] = []  # Create the list of images

    def sfm_run(self, f_src: str, fp_method=FM_AKAZE):
        """
        Main SFM routine.
        :param f_src: The folder source which contains the images
        :param fp_method: The feature point extraction method
        :return: Nothing
        """
        self.sfm_set_image_list(f_src)

    def sfm_set_image_list(self, f_src):
        """
        Append images to list.
        :param f_src: The folder which containes the images
        :return: Nothing
        """
        for suffix in self.ALL_SUPPORTED_FORMATS_LIST:  # For all supported suffixes
            imgs_src = os.path.normpath(f_src) + os.path.normpath("/" + suffix)  # Set the searching path
            img_files_path = glob.glob(imgs_src)  # Take all image paths with selected suffix
            for path in img_files_path:  # For all paths in img_file_path
                img_tmp = Image()  # Create temporary Image() instance
                img_tmp.img_open_image(path)  # Open the image (write the path information)
                self.image_list.append(img_tmp)  # Append image to list
                # print(img_tmp.IMG_NAME())

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        # print(len(self.image_list))
        # for img in self.image_list:
            # print(img.IMG_NAME())
        # print(self.image_list[0].CAMERA_MATRIX())
        # print(self.image_list[0].INFO())
        # -------------------------------------------- #
