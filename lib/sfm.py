# Python Libraries #
import os
import glob
import math as mth
from sklearn.cluster import dbscan

# Written Libraries #
from lib.image import *
from lib.image_matching import *
from lib.global_functions import *
from lib.rigid_transform_3D import *

FAST_MATCH = 0
FAST_MEDIUM_MATCH = 1
MEDIUM_SPEED_MATCH = 2
SLOW_MATCH = 3

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

        self.model_id: [] = []  # The id list for each model point [id]
        self.model_points: [] = []  # The list of model points [X, Y, Z]
        self.model_colors: [] = []  # The list of the colors for each point [R, G, B]

    def sfm_run(self, f_src: str, fp_method=FM_AKAZE, set_camera_method=CAM_APPROXIMATE,
                match_method=MATCH_FLANN, speed_match=MEDIUM_SPEED_MATCH, set_quality=Q_HIGH,
                camera_approximate_method=APPROXIMATE_WIDTH_HEIGHT):
        """
        Main SFM routine.
        :param f_src: The folder source which contains the images
        :param fp_method: The feature point extraction method
        :param set_camera_method: The method for calculate camera parameters
        :param match_method: The method which will be performed the matching
        :param speed_match: The matching way
        :param set_quality: The maximum pixel size the image needs to have for the algorithm to run
        :param camera_approximate_method: The way the focal length will be approximated
        :return: Nothing
        """
        self.sfm_set_image_list(f_src)  # Open Images
        self.sfm_find_feature_points(flag=fp_method, set_camera_method=set_camera_method,
                                     set_quality=set_quality,
                                     camera_approximate_method=camera_approximate_method)  # Find feature points
        if speed_match == FAST_MATCH:
            self.sfm_image_matching_fast(match_method=match_method)  # Match Images
        elif speed_match == FAST_MEDIUM_MATCH:
            self.sfm_image_matching_fast_medium(match_method=match_method)  # Match Images
        elif speed_match == SLOW_MATCH:
            self.sfm_image_matching_slow(match_method=match_method)  # Match Images
        else:
            self.sfm_image_matching_medium(match_method=match_method)  # Match Images
        self.sfm_model_creation_dictionary()
        self.sfm_remove_noise_from_model()

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
            img_files_path.sort()  # Sort the list
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

    def sfm_find_feature_points(self, flag=FM_AKAZE, set_camera_method=CAM_APPROXIMATE, set_quality=Q_HIGH,
                                camera_approximate_method=APPROXIMATE_WIDTH_HEIGHT):
        """
        The routine for finding the feature points.
        :param flag: Feature points flag
        :param set_camera_method: Camera setting method
        :param set_quality: The maximum pixel size the image needs to have for the algorithm to run
        :param camera_approximate_method: The way the focal length will be approximated
        :return: Nothing
        """
        for img in self.image_list:  # For all images in image list
            message_print("FIND FEATURE POINTS FOR IMAGE: %s" % img.IMG_NAME())  # Console Message
            img.img_find_features(flag=flag, set_camera_method=set_camera_method,
                                  set_quality=set_quality,
                                  camera_approximate_method=camera_approximate_method)  # Find Feature Method
            message_print(img.INFO())  # Print image information
            # print(img.INFO_DEBUGGING())  # Print debugging image information

    def sfm_image_matching_fast(self, match_method=MATCH_FLANN):
        """
        The routine for image matching.
        :return: Nothing
        """
        image_list_size = len(self.image_list)  # Take the size of the image list (the number of images in list)
        matchSize = image_list_size - 1  # Set a matchSize counter
        loop_counter = 1  # Create a loop counter for message printing
        match_id_counter = 0  # Create a counter for keeping track the match id
        for index_L in range(0, image_list_size-1):  # For all images which can be left (0, N-1)
            index_R = index_L+1
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
        print("")
        message_print("Found %d matches." % len(self.match_list))  # Console Messaging

    def sfm_image_matching_fast_medium(self, match_method=MATCH_FLANN):
        """
        The routine for image matching.
        :return: Nothing
        """
        image_list_size = len(self.image_list)  # Take the size of the image list (the number of images in list)
        matchSize = image_list_size - 1  # Set a matchSize counter
        loop_counter = 1  # Create a loop counter for message printing
        match_id_counter = 0  # Create a counter for keeping track the match id
        for index_L in range(0, image_list_size-1):  # For all images which can be left (0, N-1)
            for index_R in range(index_L + 1, image_list_size):  # For all images which can be right (1, N)
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
                        break
                else:  # Else run flann matching
                    if m_img.m_img_match_images_flann():  # If images can be matched
                        self.match_list.append(m_img)  # Append the match to list
                        match_id_counter += 1  # Increment the match id counter
                        break
            loop_counter += 1  # Increment the loop counter
        print("")
        message_print("Found %d matches." % len(self.match_list))  # Console Messaging

    def sfm_image_matching_medium(self, match_method=MATCH_FLANN):
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
        print("")
        message_print("Found %d matches." % len(self.match_list))  # Console Messaging

    def sfm_image_matching_slow(self, match_method=MATCH_FLANN):
        """
        The routine for image matching.
        :return: Nothing
        """
        image_list_size = len(self.image_list)  # Take the size of the image list (the number of images in list)
        matchSize = image_list_size * image_list_size  # Perform the previous equation
        loop_counter = 1  # Create a loop counter for message printing
        match_id_counter = 0  # Create a counter for keeping track the match id
        for index_L in range(0, image_list_size):  # For all images which can be left (0, N-1)
            for index_R in range(0, image_list_size):
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
        print("")
        message_print("Found %d matches." % len(self.match_list))  # Console Messaging

    def sfm_model_creation_dictionary(self):
        print("")
        message_print("Create Final Model")
        message_print("Find First Model")

        model_curr_points = self.match_list[0].MODEL_POINTS_LIST()  # The list of model points
        model_curr_colors = self.match_list[0].MODEL_COLOR_LIST()  # The list of corresponding colors

        model_points = []
        model_colors = []
        model_ids = []  # the list with all ids that refers to the same point

        model_ids_tmp = self.match_list[0].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
        model_curr_image_L: Image = self.match_list[0].IMG_LEFT()  # Set the current left image
        model_curr_image_R: Image = self.match_list[0].IMG_RIGHT()  # Set the current right image
        model_curr_size = len(self.match_list[0].MODEL_POINTS_LIST())  # Find the size of the current model
        image_list_size = len(self.image_list)
        for i in range(0, image_list_size):
            tmp = {}
            model_ids.append(tmp)

        for i in range(0, model_curr_size):
            model_points.append(model_curr_points[i])
            model_colors.append(model_curr_colors[i])

            model_ids[model_curr_image_L.IMG_ID()][str(model_ids_tmp[i][0])] = i
            model_ids[model_curr_image_R.IMG_ID()][str(model_ids_tmp[i][1])] = i

        # print(model_ids)
        match_list_size = len(self.match_list)
        show_size = match_list_size - 1
        for i in range(1, match_list_size):
            model_ids_tmp = self.match_list[i].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
            model_curr_image_L = self.match_list[i].IMG_LEFT()  # Set the current left image
            model_curr_image_R = self.match_list[i].IMG_RIGHT()  # Set the current right image
            model_curr_size = len(self.match_list[i].MODEL_POINTS_LIST())  # Find the size of the current model
            model_curr_points = self.match_list[i].MODEL_POINTS_LIST()
            model_curr_colors = self.match_list[i].MODEL_COLOR_LIST()

            print("")
            print("(%d / " % i + "%d)" % show_size)
            message_print("Add Model %s - " % model_curr_image_L.IMG_NAME() +
                          "%s to Model" % model_curr_image_R.IMG_NAME())

            model_fin_m_ids = []
            model_fin_m_points = []
            model_pair_m_ids = []
            model_pair_m_points = []
            model_fin_pair_m_ids = []
            for j in range(0, model_curr_size):
                if model_ids[model_curr_image_L.IMG_ID()].get(str(model_ids_tmp[j][0]), -1) != -1:
                    m = model_ids[model_curr_image_L.IMG_ID()].get(str(model_ids_tmp[j][0]), -1)
                    model_fin_m_ids.append(m)
                    model_pair_m_ids.append(j)
                    p_tmp_fin = model_points[m]
                    p_tmp_curr = model_curr_points[j]
                    model_fin_m_points.append(p_tmp_fin)
                    model_pair_m_points.append(p_tmp_curr)
                    tmp = [m, j]
                    model_fin_pair_m_ids.append(tmp)
                elif model_ids[model_curr_image_R.IMG_ID()].get(str(model_ids_tmp[j][1]), -1) != -1:
                    m = model_ids[model_curr_image_R.IMG_ID()].get(str(model_ids_tmp[j][1]), -1)
                    model_fin_m_ids.append(m)
                    model_pair_m_ids.append(j)
                    p_tmp_fin = model_points[m]
                    p_tmp_curr = model_curr_points[j]
                    model_fin_m_points.append(p_tmp_fin)
                    model_pair_m_points.append(p_tmp_curr)
                    tmp = [m, j]
                    model_fin_pair_m_ids.append(tmp)
                else:
                    tmp = [-1, -1]
                    model_fin_pair_m_ids.append(tmp)

            model_matching_size = len(model_fin_m_ids)
            if model_matching_size < 5:
                print("Model cannot be added due to few corresponding points.")
            else:
                # print(model_fin_m_ids)
                # print(model_pair_m_ids)
                # print(len(model_fin_m_ids))
                # print(len(model_pair_m_ids))

                # Find Scale
                scale, scale_error = find_scale_parameter(model_fin_m_points, model_pair_m_points)

                message_print("Scale Pair Model:")
                message_print("Scale = %f" % scale)
                message_print("Scale Error = %f" % scale_error)

                """
                X_o_prev = 0.0
                Y_o_prev = 0.0
                Z_o_prev = 0.0

                X_o_new = 0.0
                Y_o_new = 0.0
                Z_o_new = 0.0
                """

                # Scale Current model
                model_curr_scaled_points = []
                for j in range(0, model_curr_size):
                    """
                    X_o_prev += model_curr_points[j][0]
                    Y_o_prev += model_curr_points[j][1]
                    Z_o_prev += model_curr_points[j][2]
                    """

                    x = model_curr_points[j][0] * scale
                    y = model_curr_points[j][1] * scale
                    z = model_curr_points[j][2] * scale
                    tmp = [x, y, z]
                    model_curr_scaled_points.append(tmp)

                    """
                    X_o_new += x
                    Y_o_new += y
                    Z_o_new += z
                X_o_prev /= model_curr_size
                Y_o_prev /= model_curr_size
                Z_o_prev /= model_curr_size
                X_o_new /= model_curr_size
                Y_o_new /= model_curr_size
                Z_o_new /= model_curr_size

                dx = X_o_prev - X_o_new
                dy = Y_o_prev - Y_o_new
                dz = Z_o_prev - Z_o_new

                for j in range(0, model_curr_size):
                    model_curr_scaled_points[j][0] += dx
                    model_curr_scaled_points[j][1] += dy
                    model_curr_scaled_points[j][2] += dz
                """
                # print(len(model_pair_m_points))
                # print(model_pair_m_points)
                for j in range(0, model_matching_size):
                    model_pair_m_points[j] = model_curr_scaled_points[model_pair_m_ids[j]]

                # print(model_pair_m_points)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                # export_path = os.path.expanduser("~/Desktop")
                # export_path += "/sfm_tmp/scaled"
                # export_path_norm = os.path.normpath(export_path)
                # if not os.path.exists(export_path_norm):
                #    os.mkdir(export_path_norm)
                # export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                #               + model_curr_image_R.IMG_NAME() + "_scaled.ply"
                # export_path_norm = os.path.normpath(export_path)
                # export_as_ply(model_curr_scaled_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #

                R, t = rigid_transform_3D(np.transpose(model_pair_m_points), np.transpose(model_fin_m_points))

                message_print("Calculate Rotation & Translation Matrices:")
                message_print("Rotation = ")
                print(R)
                message_print("Translation = ")
                print(t)

                # print(model_curr_scaled_points)
                A = np.transpose(model_curr_scaled_points)
                m, n = A.shape
                B2 = np.dot(R, A) + np.tile(t, (1, n))
                model_curr_scaled_R_t_points = np.transpose(B2)
                # print(model_curr_scaled_R_t_points)
                model_curr_scaled_points.clear()

                # -------------------------------------------- #
                #export_path = os.path.expanduser("~/Desktop")
                #export_path += "/sfm_tmp/"
                #export_path_norm = os.path.normpath(export_path)
                #if not os.path.exists(export_path_norm):
                #    os.mkdir(export_path_norm)
                #export_path += "/final"
                #export_path_norm = os.path.normpath(export_path)
                #if not os.path.exists(export_path_norm):
                #    os.mkdir(export_path_norm)
                #export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                #               + model_curr_image_R.IMG_NAME() + "_final.ply"
                #export_path_norm = os.path.normpath(export_path)
                #export_as_ply(model_curr_scaled_R_t_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #
                model_fin_m_ids.clear()
                model_fin_m_points.clear()
                model_pair_m_ids.clear()
                model_pair_m_points.clear()

                for j in range(0, model_curr_size):
                    if model_fin_pair_m_ids[j][0] == -1:
                        model_points.append(model_curr_scaled_R_t_points[j])
                        model_colors.append(model_curr_colors[j])
                        index = len(model_points) - 1
                        model_ids[model_curr_image_L.IMG_ID()][str(model_ids_tmp[j][0])] = index
                        model_ids[model_curr_image_R.IMG_ID()][str(model_ids_tmp[j][1])] = index
                    else:
                        k_ind = model_fin_pair_m_ids[j][0]
                        l_ind = model_fin_pair_m_ids[j][1]
                        model_points[k_ind][0] += model_curr_scaled_R_t_points[l_ind][0]
                        model_points[k_ind][1] += model_curr_scaled_R_t_points[l_ind][1]
                        model_points[k_ind][2] += model_curr_scaled_R_t_points[l_ind][2]

                        model_colors[k_ind][0] += model_curr_colors[l_ind][0]
                        model_colors[k_ind][1] += model_curr_colors[l_ind][1]
                        model_colors[k_ind][2] += model_curr_colors[l_ind][2]

                        model_points[k_ind][0] /= 2
                        model_points[k_ind][1] /= 2
                        model_points[k_ind][2] /= 2

                        model_colors[k_ind][0] /= 2
                        model_colors[k_ind][1] /= 2
                        model_colors[k_ind][2] /= 2

                # print(model_ids)

                model_size = len(model_points)
                model_points_T = np.transpose(model_points)
                model_centroid = [np.mean(model_points_T[0]), np.mean(model_points_T[1]), np.mean(model_points_T[2])]
                model_centroid_err = [np.std(model_points_T[0]), np.std(model_points_T[1]), np.std(model_points_T[2])]

                message_print("New model size = %d" % model_size)
                message_print("New model centroid = ")
                print(model_centroid)
                message_print("New model centroid errors = ")
                print(model_centroid_err)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                export_path = os.path.expanduser("~/Desktop")
                export_path += "/sfm_tmp/"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/final"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/model_" + str(i) + "_final.ply"
                export_path_norm = os.path.normpath(export_path)
                export_as_ply(model_points, model_colors, export_path_norm)
                # -------------------------------------------- #

        self.model_points = model_points
        self.model_colors = model_colors
        model_size = len(self.model_points)
        for i in range(0, model_size):
            self.model_id.append(i)

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        export_path = os.path.expanduser("~/Desktop")
        export_path += "/sfm_tmp/"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model_final.ply"
        export_path_norm = os.path.normpath(export_path)
        export_as_ply(self.model_points, self.model_colors, export_path_norm)
        # -------------------------------------------- #

    def sfm_model_creation_fast(self):
        print("")
        message_print("Create Final Model")
        message_print("Find First Model")

        model_curr_points = self.match_list[0].MODEL_POINTS_LIST()  # The list of model points
        model_curr_colors = self.match_list[0].MODEL_COLOR_LIST()  # The list of corresponding colors

        model_points = []
        model_colors = []

        model_ids = []  # the list with all ids that refers to the same point

        model_ids_tmp = self.match_list[0].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
        model_curr_image_L: Image = self.match_list[0].IMG_LEFT()  # Set the current left image
        model_curr_image_R: Image = self.match_list[0].IMG_RIGHT()  # Set the current right image
        model_curr_size = len(self.match_list[0].MODEL_POINTS_LIST())  # Find the size of the current model
        image_list_size = len(self.image_list)
        for i in range(0, image_list_size):
            tmp = []
            model_ids.append(tmp)

        for i in range(0, model_curr_size):
            model_points.append(model_curr_points[i])
            model_colors.append(model_curr_colors[i])
            """
            new_entry = self.sfm_new_entry()
            model_ids.append(new_entry)
            model_ids[i][model_curr_image_L.IMG_ID()] = model_ids_tmp[i][0]
            model_ids[i][model_curr_image_R.IMG_ID()] = model_ids_tmp[i][1]
            """
            tmp_L = [model_ids_tmp[i][0], i]
            tmp_R = [model_ids_tmp[i][1], i]
            model_ids[model_curr_image_L.IMG_ID()].append(tmp_L)
            model_ids[model_curr_image_R.IMG_ID()].append(tmp_R)

        # print(model_ids)
        match_list_size = len(self.match_list)
        show_size = match_list_size - 1
        for i in range(1, match_list_size):
            model_ids_tmp = self.match_list[i].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
            model_curr_image_L = self.match_list[i].IMG_LEFT()  # Set the current left image
            model_curr_image_R = self.match_list[i].IMG_RIGHT()  # Set the current right image
            model_curr_size = len(self.match_list[i].MODEL_POINTS_LIST())  # Find the size of the current model
            model_curr_points = self.match_list[i].MODEL_POINTS_LIST()
            model_curr_colors = self.match_list[i].MODEL_COLOR_LIST()

            print("")
            print("(%d / " % i + "%d)" % show_size)
            message_print("Add Model %s - " % model_curr_image_L.IMG_NAME() +
                          "%s to Model" % model_curr_image_R.IMG_NAME())

            model_fin_m_ids = []
            model_fin_m_points = []
            model_pair_m_ids = []
            model_pair_m_points = []
            model_fin_pair_m_ids = []

            curr_left_ids = len(model_ids[model_curr_image_L.IMG_ID()])
            curr_right_ids = len(model_ids[model_curr_image_R.IMG_ID()])
            for j in range(0, model_curr_size):
                is_match = False
                for k in range(0, curr_left_ids):
                    if model_ids[model_curr_image_L.IMG_ID()][k][0] == model_ids_tmp[j][0]:
                        m = model_ids[model_curr_image_L.IMG_ID()][k][1]
                        model_fin_m_ids.append(m)
                        model_pair_m_ids.append(j)
                        p_tmp_fin = model_points[m]
                        p_tmp_curr = model_curr_points[j]
                        model_fin_m_points.append(p_tmp_fin)
                        model_pair_m_points.append(p_tmp_curr)
                        tmp = [m, j]
                        model_fin_pair_m_ids.append(tmp)
                        is_match = True
                        break
                if not is_match:
                    for k in range(0, curr_right_ids):
                        if model_ids[model_curr_image_R.IMG_ID()][k][0] == model_ids_tmp[j][1]:
                            m = model_ids[model_curr_image_R.IMG_ID()][k][1]
                            model_fin_m_ids.append(m)
                            model_pair_m_ids.append(j)
                            p_tmp_fin = model_points[m]
                            p_tmp_curr = model_curr_points[j]
                            model_fin_m_points.append(p_tmp_fin)
                            model_pair_m_points.append(p_tmp_curr)
                            tmp = [m, j]
                            model_fin_pair_m_ids.append(tmp)
                            is_match = True
                            break
                if not is_match:
                    tmp = [-1, -1]
                    model_fin_pair_m_ids.append(tmp)

            model_matching_size = len(model_fin_m_ids)
            if model_matching_size < 5:
                print("Model cannot be added due to few corresponding points.")
            else:
                # print(model_fin_m_ids)
                # print(model_pair_m_ids)
                # print(len(model_fin_m_ids))
                # print(len(model_pair_m_ids))

                # Find Scale
                scale, scale_error = find_scale_parameter(model_fin_m_points, model_pair_m_points)

                message_print("Scale Pair Model:")
                message_print("Scale = %f" % scale)
                message_print("Scale Error = %f" % scale_error)

                """
                X_o_prev = 0.0
                Y_o_prev = 0.0
                Z_o_prev = 0.0
    
                X_o_new = 0.0
                Y_o_new = 0.0
                Z_o_new = 0.0
                """

                # Scale Current model
                model_curr_scaled_points = []
                for j in range(0, model_curr_size):
                    """
                    X_o_prev += model_curr_points[j][0]
                    Y_o_prev += model_curr_points[j][1]
                    Z_o_prev += model_curr_points[j][2]
                    """

                    x = model_curr_points[j][0] * scale
                    y = model_curr_points[j][1] * scale
                    z = model_curr_points[j][2] * scale
                    tmp = [x, y, z]
                    model_curr_scaled_points.append(tmp)

                    """
                    X_o_new += x
                    Y_o_new += y
                    Z_o_new += z
                X_o_prev /= model_curr_size
                Y_o_prev /= model_curr_size
                Z_o_prev /= model_curr_size
                X_o_new /= model_curr_size
                Y_o_new /= model_curr_size
                Z_o_new /= model_curr_size
                
                dx = X_o_prev - X_o_new
                dy = Y_o_prev - Y_o_new
                dz = Z_o_prev - Z_o_new
    
                for j in range(0, model_curr_size):
                    model_curr_scaled_points[j][0] += dx
                    model_curr_scaled_points[j][1] += dy
                    model_curr_scaled_points[j][2] += dz
                """
                # print(len(model_pair_m_points))
                # print(model_pair_m_points)
                for j in range(0, model_matching_size):
                    model_pair_m_points[j] = model_curr_scaled_points[model_pair_m_ids[j]]
                # print(model_pair_m_points)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                # export_path = os.path.expanduser("~/Desktop")
                # export_path += "/sfm_tmp/scaled"
                # export_path_norm = os.path.normpath(export_path)
                # if not os.path.exists(export_path_norm):
                #    os.mkdir(export_path_norm)
                # export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                #               + model_curr_image_R.IMG_NAME() + "_scaled.ply"
                # export_path_norm = os.path.normpath(export_path)
                # export_as_ply(model_curr_scaled_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #

                R, t = rigid_transform_3D(np.transpose(model_pair_m_points), np.transpose(model_fin_m_points))

                message_print("Calculate Rotation & Translation Matrices:")
                message_print("Rotation = ")
                print(R)
                message_print("Translation = ")
                print(t)

                # print(model_curr_scaled_points)
                A = np.transpose(model_curr_scaled_points)
                m, n = A.shape
                B2 = np.dot(R, A) + np.tile(t, (1, n))
                model_curr_scaled_R_t_points = np.transpose(B2)
                # print(model_curr_scaled_R_t_points)
                model_curr_scaled_points.clear()

                # -------------------------------------------- #
                export_path = os.path.expanduser("~/Desktop")
                export_path += "/sfm_tmp/"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/final"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                               + model_curr_image_R.IMG_NAME() + "_final.ply"
                export_path_norm = os.path.normpath(export_path)
                export_as_ply(model_curr_scaled_R_t_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #
                model_fin_m_ids.clear()
                model_fin_m_points.clear()
                model_pair_m_ids.clear()
                model_pair_m_points.clear()

                for j in range(0, model_curr_size):
                    if model_fin_pair_m_ids[j][0] == -1:
                        model_points.append(model_curr_scaled_R_t_points[j])
                        model_colors.append(model_curr_colors[j])
                        """
                        new_entry = self.sfm_new_entry()
                        model_ids.append(new_entry)
                        index = len(model_ids) - 1
                        model_ids[index][model_curr_image_L.IMG_ID()] = model_ids_tmp[j][0]
                        model_ids[index][model_curr_image_R.IMG_ID()] = model_ids_tmp[j][1]
                        """
                        index = len(model_points) - 1
                        tmp_L = [model_ids_tmp[j][0], index]
                        tmp_R = [model_ids_tmp[j][1], index]
                        model_ids[model_curr_image_L.IMG_ID()].append(tmp_L)
                        model_ids[model_curr_image_R.IMG_ID()].append(tmp_R)
                    else:
                        k_ind = model_fin_pair_m_ids[j][0]
                        l_ind = model_fin_pair_m_ids[j][1]
                        model_points[k_ind][0] += model_curr_scaled_R_t_points[l_ind][0]
                        model_points[k_ind][1] += model_curr_scaled_R_t_points[l_ind][1]
                        model_points[k_ind][2] += model_curr_scaled_R_t_points[l_ind][2]

                        model_colors[k_ind][0] += model_curr_colors[l_ind][0]
                        model_colors[k_ind][1] += model_curr_colors[l_ind][1]
                        model_colors[k_ind][2] += model_curr_colors[l_ind][2]

                        model_points[k_ind][0] /= 2
                        model_points[k_ind][1] /= 2
                        model_points[k_ind][2] /= 2

                        model_colors[k_ind][0] /= 2
                        model_colors[k_ind][1] /= 2
                        model_colors[k_ind][2] /= 2

                # print(model_ids)

                model_size = len(model_points)
                model_points_T = np.transpose(model_points)
                model_centroid = [np.mean(model_points_T[0]), np.mean(model_points_T[1]), np.mean(model_points_T[2])]
                model_centroid_err = [np.std(model_points_T[0]), np.std(model_points_T[1]), np.std(model_points_T[2])]

                message_print("New model size = %d" % model_size)
                message_print("New model centroid = ")
                print(model_centroid)
                message_print("New model centroid errors = ")
                print(model_centroid_err)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                export_path = os.path.expanduser("~/Desktop")
                export_path += "/sfm_tmp/"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/final"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/model_" + str(i) + "_final.ply"
                export_path_norm = os.path.normpath(export_path)
                export_as_ply(model_points, model_colors, export_path_norm)
                # -------------------------------------------- #

        self.model_points = model_points
        self.model_colors = model_colors
        model_size = len(self.model_points)
        for i in range(0, model_size):
            self.model_id.append(i)

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        export_path = os.path.expanduser("~/Desktop")
        export_path += "/sfm_tmp/"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model_final.ply"
        export_path_norm = os.path.normpath(export_path)
        export_as_ply(self.model_points, self.model_colors, export_path_norm)
        # -------------------------------------------- #

    def sfm_model_creation_slow(self):
        print("")
        message_print("Create Final Model")
        message_print("Find First Model")

        model_curr_points = self.match_list[0].MODEL_POINTS_LIST()  # The list of model points
        model_curr_colors = self.match_list[0].MODEL_COLOR_LIST()  # The list of corresponding colors

        model_points = []
        model_colors = []

        model_ids = []  # the list with all ids that refers to the same point

        model_ids_tmp = self.match_list[0].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
        model_curr_image_L: Image = self.match_list[0].IMG_LEFT()  # Set the current left image
        model_curr_image_R: Image = self.match_list[0].IMG_RIGHT()  # Set the current right image
        model_curr_size = len(self.match_list[0].MODEL_POINTS_LIST())  # Find the size of the current model
        image_list_size = len(self.image_list)
        for i in range(0, image_list_size):
            tmp = []
            model_ids.append(tmp)

        for i in range(0, model_curr_size):
            model_points.append(model_curr_points[i])
            model_colors.append(model_curr_colors[i])

            new_entry = self.sfm_new_entry()
            model_ids.append(new_entry)
            model_ids[i][model_curr_image_L.IMG_ID()] = model_ids_tmp[i][0]
            model_ids[i][model_curr_image_R.IMG_ID()] = model_ids_tmp[i][1]

        # print(model_ids)
        match_list_size = len(self.match_list)
        show_size = match_list_size - 1
        for i in range(1, match_list_size):
            model_ids_tmp = self.match_list[i].MODEL_ID_LIST()  # Set the point list of the current model to a tmp list
            model_curr_image_L = self.match_list[i].IMG_LEFT()  # Set the current left image
            model_curr_image_R = self.match_list[i].IMG_RIGHT()  # Set the current right image
            model_curr_size = len(self.match_list[i].MODEL_POINTS_LIST())  # Find the size of the current model
            model_curr_points = self.match_list[i].MODEL_POINTS_LIST()
            model_curr_colors = self.match_list[i].MODEL_COLOR_LIST()
            model_fin_size = len(model_ids)

            print("")
            print("(%d / " % i + "%d)" % show_size)
            message_print("Add Model %s - " % model_curr_image_L.IMG_NAME() +
                          "%s to Model" % model_curr_image_R.IMG_NAME())

            model_fin_m_ids = []
            model_fin_m_points = []
            model_pair_m_ids = []
            model_pair_m_points = []
            model_fin_pair_m_ids = []

            for j in range(0, model_curr_size):
                is_not_match = False
                for k in range(0, model_fin_size):
                    if model_ids[k][model_curr_image_L.IMG_ID()] == model_ids_tmp[j][0]:
                        model_fin_m_ids.append(k)
                        model_pair_m_ids.append(j)
                        p_tmp_fin = model_points[k]
                        p_tmp_curr = model_curr_points[j]
                        model_fin_m_points.append(p_tmp_fin)
                        model_pair_m_points.append(p_tmp_curr)
                        tmp = [k, j]
                        model_fin_pair_m_ids.append(tmp)
                        is_not_match = False
                        break
                    elif model_ids[k][model_curr_image_R.IMG_ID()] == model_ids_tmp[j][1]:
                        model_fin_m_ids.append(k)
                        model_pair_m_ids.append(j)
                        p_tmp_fin = model_points[k]
                        p_tmp_curr = model_curr_points[j]
                        model_fin_m_points.append(p_tmp_fin)
                        model_pair_m_points.append(p_tmp_curr)
                        tmp = [k, j]
                        model_fin_pair_m_ids.append(tmp)
                        is_not_match = False
                        break
                    else:
                        is_not_match = True
                if is_not_match:
                    tmp = [-1, -1]
                    model_fin_pair_m_ids.append(tmp)

            model_matching_size = len(model_fin_m_ids)
            if model_matching_size < 5:
                print("Model cannot be added due to few corresponding points.")
            else:
                # print(model_fin_m_ids)
                # print(model_pair_m_ids)
                print(len(model_fin_m_ids))
                print(len(model_pair_m_ids))

                # Find Scale
                scale, scale_error = find_scale_parameter(model_fin_m_points, model_pair_m_points)

                message_print("Scale Pair Model:")
                message_print("Scale = %f" % scale)
                message_print("Scale Error = %f" % scale_error)

                """
                X_o_prev = 0.0
                Y_o_prev = 0.0
                Z_o_prev = 0.0

                X_o_new = 0.0
                Y_o_new = 0.0
                Z_o_new = 0.0
                """

                # Scale Current model
                model_curr_scaled_points = []
                for j in range(0, model_curr_size):
                    """
                    X_o_prev += model_curr_points[j][0]
                    Y_o_prev += model_curr_points[j][1]
                    Z_o_prev += model_curr_points[j][2]
                    """

                    x = model_curr_points[j][0] * scale
                    y = model_curr_points[j][1] * scale
                    z = model_curr_points[j][2] * scale
                    tmp = [x, y, z]
                    model_curr_scaled_points.append(tmp)

                    """
                    X_o_new += x
                    Y_o_new += y
                    Z_o_new += z
                X_o_prev /= model_curr_size
                Y_o_prev /= model_curr_size
                Z_o_prev /= model_curr_size
                X_o_new /= model_curr_size
                Y_o_new /= model_curr_size
                Z_o_new /= model_curr_size

                dx = X_o_prev - X_o_new
                dy = Y_o_prev - Y_o_new
                dz = Z_o_prev - Z_o_new

                for j in range(0, model_curr_size):
                    model_curr_scaled_points[j][0] += dx
                    model_curr_scaled_points[j][1] += dy
                    model_curr_scaled_points[j][2] += dz
                """
                # print(len(model_pair_m_points))
                # print(model_pair_m_points)
                for j in range(0, model_matching_size):
                    model_pair_m_points[j] = model_curr_scaled_points[model_pair_m_ids[j]]
                # print(model_pair_m_points)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                # export_path = os.path.expanduser("~/Desktop")
                # export_path += "/sfm_tmp/scaled"
                # export_path_norm = os.path.normpath(export_path)
                # if not os.path.exists(export_path_norm):
                #    os.mkdir(export_path_norm)
                # export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                #               + model_curr_image_R.IMG_NAME() + "_scaled.ply"
                # export_path_norm = os.path.normpath(export_path)
                # export_as_ply(model_curr_scaled_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #

                R, t = rigid_transform_3D(np.transpose(model_pair_m_points), np.transpose(model_fin_m_points))

                message_print("Calculate Rotation & Translation Matrices:")
                message_print("Rotation = ")
                print(R)
                message_print("Translation = ")
                print(t)

                # print(model_curr_scaled_points)
                A = np.transpose(model_curr_scaled_points)
                m, n = A.shape
                B2 = np.dot(R, A) + np.tile(t, (1, n))
                model_curr_scaled_R_t_points = np.transpose(B2)
                # print(model_curr_scaled_R_t_points)
                model_curr_scaled_points.clear()

                # -------------------------------------------- #
                export_path = os.path.expanduser("~/Desktop")
                export_path += "/sfm_tmp/"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/final"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/" + model_curr_image_L.IMG_NAME() + "_" \
                               + model_curr_image_R.IMG_NAME() + "_final.ply"
                export_path_norm = os.path.normpath(export_path)
                export_as_ply(model_curr_scaled_R_t_points, model_curr_colors, export_path_norm)
                # -------------------------------------------- #
                model_fin_m_ids.clear()
                model_fin_m_points.clear()
                model_pair_m_ids.clear()
                model_pair_m_points.clear()

                for j in range(0, model_curr_size):
                    if model_fin_pair_m_ids[j][0] == -1:
                        model_points.append(model_curr_scaled_R_t_points[j])
                        model_colors.append(model_curr_colors[j])
                        new_entry = self.sfm_new_entry()
                        model_ids.append(new_entry)
                        index = len(model_ids) - 1
                        model_ids[index][model_curr_image_L.IMG_ID()] = model_ids_tmp[j][0]
                        model_ids[index][model_curr_image_R.IMG_ID()] = model_ids_tmp[j][1]
                    else:
                        k_ind = model_fin_pair_m_ids[j][0]
                        l_ind = model_fin_pair_m_ids[j][1]
                        model_points[k_ind][0] += model_curr_scaled_R_t_points[l_ind][0]
                        model_points[k_ind][1] += model_curr_scaled_R_t_points[l_ind][1]
                        model_points[k_ind][2] += model_curr_scaled_R_t_points[l_ind][2]

                        model_colors[k_ind][0] += model_curr_colors[l_ind][0]
                        model_colors[k_ind][1] += model_curr_colors[l_ind][1]
                        model_colors[k_ind][2] += model_curr_colors[l_ind][2]

                        model_points[k_ind][0] /= 2
                        model_points[k_ind][1] /= 2
                        model_points[k_ind][2] /= 2

                        model_colors[k_ind][0] /= 2
                        model_colors[k_ind][1] /= 2
                        model_colors[k_ind][2] /= 2

                model_size = len(model_points)
                model_points_T = np.transpose(model_points)
                model_centroid = [np.mean(model_points_T[0]), np.mean(model_points_T[1]), np.mean(model_points_T[2])]
                model_centroid_err = [np.std(model_points_T[0]), np.std(model_points_T[1]), np.std(model_points_T[2])]

                message_print("New model size = %d" % model_size)
                message_print("New model centroid = ")
                print(model_centroid)
                message_print("New model centroid errors = ")
                print(model_centroid_err)

                # -------------------------------------------- #
                # Uncomment the following lines for debugging. #
                # -------------------------------------------- #
                export_path = os.path.expanduser("~/Desktop")
                export_path += "/sfm_tmp/"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/final"
                export_path_norm = os.path.normpath(export_path)
                if not os.path.exists(export_path_norm):
                    os.mkdir(export_path_norm)
                export_path += "/model_" + str(i) + "_final.ply"
                export_path_norm = os.path.normpath(export_path)
                export_as_ply(model_points, model_colors, export_path_norm)
                # -------------------------------------------- #

        self.model_points = model_points
        self.model_colors = model_colors
        model_size = len(self.model_points)
        for i in range(0, model_size):
            self.model_id.append(i)

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        export_path = os.path.expanduser("~/Desktop")
        export_path += "/sfm_tmp/"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model_final.ply"
        export_path_norm = os.path.normpath(export_path)
        export_as_ply(self.model_points, self.model_colors, export_path_norm)
        # -------------------------------------------- #

    def sfm_remove_noise_from_model(self):
        model_size = len(self.model_points)
        message_print("Remove Noise Points.")
        min_samples = model_size / 2000
        model_clustering, label = dbscan(self.model_points, min_samples=min_samples)
        counter_id = 0

        model_points = []
        model_colors = []
        model_id = []
        for i in range(0, model_size):
            if label[i] != -1:
                model_points.append(self.model_points[i])
                model_colors.append(self.model_colors[i])
                model_id.append(counter_id)
                counter_id += 1

        message_print("Final Cloud = %d out of " % len(model_points) +
                      "%d previous model points." % model_size)

        self.model_points = model_points
        self.model_colors = model_colors
        self.model_id = model_id

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        export_path = os.path.expanduser("~/Desktop")
        export_path += "/sfm_tmp/"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/model_no_noise_final.ply"
        export_path_norm = os.path.normpath(export_path)
        export_as_ply(self.model_points, self.model_colors, export_path_norm)
        # -------------------------------------------- #

    def sfm_new_entry(self):
        model_new_entry_id = []  # Create an id list for the new entry points
        image_list_size = len(self.image_list)  # Take the size of the block (all opened images)
        for i in range(0, image_list_size):  # For each image
            model_new_entry_id.append(-1)  # Append -1 (-1 represent no matching)
        return model_new_entry_id
