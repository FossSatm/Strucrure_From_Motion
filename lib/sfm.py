# Python Libraries #
import os
import glob

# Written Libraries #
from lib.image import *
from lib.image_matching import *
from lib.global_functions import *


class SFM:
    """
    The Structure from Motion main class. This class contains the methods for running the SFM process.
    """
    def __init__(self):
        # Create a list with all OpenCV supported image file formats
        self.ALL_SUPPORTED_FORMATS_LIST = ["*.jpg", "*.jpeg", "*.jpe", "*.png", "*.bmp",
                                           "*.tif", "*.tiff", "*.dib", "*.pbm", "*.pgm",
                                           "*.ppm", "*.sr", "*.ras"]
        self.image_list: [] = []  # Create the list of images
        self.match_list: [] = []  # Create the list for image matching

    def sfm_run(self, f_src: str, fp_method=FM_AKAZE, set_camera_method=CAM_DEFAULT):
        """
        Main SFM routine.
        :param set_camera_method:
        :param f_src: The folder source which contains the images
        :param fp_method: The feature point extraction method
        :return: Nothing
        """
        self.sfm_set_image_list(f_src)  # Open Images
        self.sfm_find_feature_points(flag=fp_method, set_camera_method=set_camera_method)  # Find feature points
        self.sfm_image_matching()  # Match Images

    def sfm_set_image_list(self, f_src):
        """
        Append images to list.
        :param f_src: The folder which contains the images
        :return: Nothing
        """
        img_id_counter = 0  # Create an image counter for id setting
        for suffix in self.ALL_SUPPORTED_FORMATS_LIST:  # For all supported suffixes
            imgs_src = os.path.normpath(f_src) + os.path.normpath("/" + suffix)  # Set the searching path
            img_files_path = glob.glob(imgs_src)  # Take all image paths with selected suffix
            for path in img_files_path:  # For all paths in img_file_path
                img_tmp = Image()  # Create temporary Image() instance
                img_tmp.img_open_image(path, img_id=img_id_counter)  # Open the image (write the path information)
                img_id_counter += 1  # Increment the counter
                self.image_list.append(img_tmp)  # Append image to list
                # print(img_tmp.IMG_NAME())  # Uncomment for debugging

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        # print(len(self.image_list))
        # for img in self.image_list:
            # print(img.IMG_NAME())
        # print(self.image_list[0].CAMERA_MATRIX())
        # print(self.image_list[0].INFO())
        # -------------------------------------------- #

    def sfm_find_feature_points(self, flag=FM_AKAZE, set_camera_method=CAM_DEFAULT):
        """
        The routine for finding the feature points.
        :param flag: Feature points flag
        :param set_camera_method: Camera setting method
        :return: Nothing
        """
        for img in self.image_list:  # For all images in image list
            message_print("FIND FEATURE POINTS FOR IMAGE: %s" % img.IMG_NAME())  # Console Message
            img.img_find_features(flag=flag, set_camera_method=set_camera_method)  # Find Feature Method
            message_print(img.INFO())  # Print image information
            # print(img.INFO_DEBUGGING())  # Print debugging image information

    def sfm_image_matching(self):
        """
        The routine for image matching.
        :return: Nothing
        """
        matchSize = 0  # Set a matchSize counter
        image_list_size = len(self.image_list)  # Take the size of the image list (the number of images in list)
        for i in range(1, image_list_size):  # For i in range(1, block_size) => matchSize = Sum_{i=1}^{N}(blockSize - i)
            matchSize += image_list_size - i  # Perform the previous equation
        loop_counter = 0  # Create a loop counter for message printing
        match_id_counter = 0  # Create a counter for keeping track the match id
        for index_L in range(0, image_list_size-1):  # For all images which can be left (0, N-1)
            for index_R in range(index_L+1, image_list_size):  # For all images which can be right (1, N)
                # Console Messaging
                print("\n")
                message_print("(%d / " % loop_counter + "%d) " % matchSize
                              + "MATCH IMAGES %s & " % self.image_list[index_L].IMG_NAME()
                              + "%s" % self.image_list[index_R].IMG_NAME())

                m_img = ImageMatching()  # Create an ImageMatching object
                # Set the images info
                m_img.m_img_set_images(self.image_list[index_L], self.image_list[index_R], match_id_counter)
                if m_img.m_img_match_images_flann():  # If images can be matched
                    self.match_list.append(m_img)  # Append the match to list
                    match_id_counter += 1  # Increment the match id counter
                loop_counter += 1  # Increment the loop counter
        # print(len(self.match_list))  # Debugging Message
