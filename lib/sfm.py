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

        self.model_id: [] = []
        self.model_points: [] = []
        self.model_colors: [] = []

    def sfm_run(self, f_src: str, fp_method=FM_AKAZE, set_camera_method=CAM_DEFAULT,
                match_method=MATCH_FLANN):
        """
        Main SFM routine.
        :param f_src: The folder source which contains the images
        :param fp_method: The feature point extraction method
        :param set_camera_method: The method for calculate camera parameters
        :param match_method: The method which will be performed the matching
        :return: Nothing
        """
        self.sfm_set_image_list(f_src)  # Open Images
        self.sfm_find_feature_points(flag=fp_method, set_camera_method=set_camera_method)  # Find feature points
        self.sfm_image_matching(match_method=match_method)  # Match Images
        self.sfm_model_creation()

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

    def sfm_image_matching(self, match_method=MATCH_FLANN):
        """
        The routine for image matching.
        :return: Nothing
        """
        matchSize = 0  # Set a matchSize counter
        image_list_size = len(self.image_list)  # Take the size of the image list (the number of images in list)
        for i in range(1, image_list_size):  # For i in range(1, block_size) => matchSize = Sum_{i=1}^{N}(blockSize - i)
            matchSize += image_list_size - i  # Perform the previous equation
        loop_counter = 1  # Create a loop counter for message printing
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
                if match_method == MATCH_BRUTEFORCE_HAMMING:  # If Bruteforce Matching
                    if m_img.m_img_match_images_bruteforce_hamming():  # If images can be matched
                        self.match_list.append(m_img)  # Append the match to list
                        match_id_counter += 1  # Increment the match id counter
                else:  # Else run flann matching
                    if m_img.m_img_match_images_flann():  # If images can be matched
                        self.match_list.append(m_img)  # Append the match to list
                        match_id_counter += 1  # Increment the match id counter
                loop_counter += 1  # Increment the loop counter
        message_print("Found %d matches." % len(self.match_list))  # Console Messaging

    def sfm_model_creation(self):
        model_ids = []  # the list with all ids that refers to the same point

        model_ids_tmp = self.match_list[0].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
        model_curr_image_L: Image = self.match_list[0].IMG_LEFT()  # Set the current left image
        model_curr_image_R: Image = self.match_list[0].IMG_RIGHT()  # Set the current right image
        model_curr_size = len(self.match_list[0].MODEL_POINTS_LIST())  # Find the size of the current model
        model_pnt_shown = []
        for i in range(0, model_curr_size):
            new_entry = self.sfm_new_entry()
            model_ids.append(new_entry)
            model_ids[i][model_curr_image_L.IMG_ID()] = model_ids_tmp[i][0]
            model_ids[i][model_curr_image_R.IMG_ID()] = model_ids_tmp[i][1]
            model_pnt_shown.append(1)

        model_points = self.match_list[0].MODEL_POINTS_LIST()  # The list of model points
        model_colors = self.match_list[0].MODEL_COLOR_LIST()  # The list of corresponding colors

        match_list_size = len(self.match_list)
        for i in range(1, match_list_size):
            model_ids_tmp = self.match_list[i].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
            model_curr_image_L = self.match_list[i].IMG_LEFT()  # Set the current left image
            model_curr_image_R = self.match_list[i].IMG_RIGHT()  # Set the current right image
            model_curr_size = len(self.match_list[i].MODEL_POINTS_LIST())  # Find the size of the current model
            model_fin_size = len(model_ids)

            model_fin_m_ids = []
            model_pair_m_ids = []
            for j in range(0, model_curr_size):
                for k in range(0, model_fin_size):
                    if model_ids[j][model_curr_image_L.IMG_ID()] == model_ids_tmp[j][0]:
                        model_fin_m_ids.append(k)
                        model_pair_m_ids.append(j)
                    elif model_ids[j][model_curr_image_R.IMG_ID()] == model_ids_tmp[j][1]:
                        model_fin_m_ids.append(k)
                        model_pair_m_ids.append(j)

    def sfm_new_entry(self):
        model_new_entry_id = []  # Create an id list for the new entry points
        image_list_size = len(self.image_list)  # Take the size of the block (all opened images)
        for i in range(0, image_list_size):  # For each image
            model_new_entry_id.append(-1)  # Append -1 (-1 represent no matching)
        return model_new_entry_id
