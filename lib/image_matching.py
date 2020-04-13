# Written Libraries #
from lib.image import *
from lib.global_functions import *

MATCH_FLANN = 0
MATCH_BRUTEFORCE_HAMMING = 1

LOWE_RATIO = 0.9


class ImageMatching:
    """
    A class which match 2 images
    """
    def __init__(self):
        self.m_img_id: int = 0
        self.imgL = Image()
        self.imgR = Image()

        self.model_coord_list: [] = []  # [X, Y, Z]
        self.model_color_list: [] = []  # [R, G, B]
        self.model_coord_id_list: [] = []  # [id_L, id_R]

    # ---------------- #
    # RETURN FUNCTIONS #
    # ---------------- #
    def IMG_MATCH_ID(self):
        return self.m_img_id

    def IMG_LEFT(self):
        return self.imgL

    def IMG_RIGHT(self):
        return self.imgR

    def MODEL_ID_LIST(self):
        return self.model_coord_id_list

    def MODEL_POINTS_LIST(self):
        return self.model_coord_list

    def MODEL_COLOR_LIST(self):
        return self.model_color_list

    # ------------------ #
    # MATCHING FUNCTIONS #
    # ------------------ #
    def m_img_set_images(self, imgL: Image, imgR: Image, m_img_id=0):
        """
        Set the matching information.
        :param imgL: The left image
        :param imgR: The right image
        :param m_img_id: The matching id
        :return: Nothing
        """
        self.m_img_id = m_img_id
        self.imgL = imgL
        self.imgR = imgR

    def m_img_match_images_bruteforce_hamming(self):
        method_brute = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        descr_L = self.imgL.DESCRIPTOR_LIST()
        descr_R = self.imgR.DESCRIPTOR_LIST()

        matches = method_brute.knnMatch(descr_L, descr_R, k=2)
        return self.m_img_match(matches=matches)

    def m_img_match_images_flann(self):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        method_flann = cv.FlannBasedMatcher(index_params, search_params)

        descr_L = self.imgL.DESCRIPTOR_LIST()
        descr_L = np.array(descr_L, dtype=np.float32)
        descr_R = self.imgR.DESCRIPTOR_LIST()
        descr_R = np.array(descr_R, dtype=np.float32)

        matches = method_flann.knnMatch(descr_L, descr_R, k=2)
        return self.m_img_match(matches=matches)

    def m_img_match(self, matches):
        kp_L = self.imgL.KEYPOINT_LIST()
        kp_R = self.imgR.KEYPOINT_LIST()
        kp_ids_L = self.imgL.KEYPOINT_IDS_LIST()
        kp_ids_R = self.imgR.KEYPOINT_IDS_LIST()

        # Find all good points as per Lower's ratio.
        good_matches = []  # list for good matches
        points_L_img = []  # list for the points coords in the left image
        points_R_img = []  # list for the points coords in the right image
        points_L_img_ids = []  # list for the ids of the left image
        points_R_img_ids = []  # list for the ids of the right image
        match_pnt_size = 0  # counter for the match point size
        for m, n in matches:
            match_pnt_size += 1  # increase the counter (this counter is used for console debugging)
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
                points_L_img.append(kp_L[m.queryIdx].pt)  # Take p_coords for left img
                points_R_img.append(kp_R[m.trainIdx].pt)  # Take p_coords for right img

                points_L_img_ids.append(kp_ids_L[m.queryIdx])  # Take the ids for the left image
                points_R_img_ids.append(kp_ids_R[m.trainIdx])  # Take the ids for the right image
        g_points_size = len(good_matches)  # take the size of good matches

        message_print("Found %d good matches out of " % g_points_size + " %d matches." % match_pnt_size)
        if g_points_size < 0.25 * match_pnt_size:
            message_print("Not a good match!")
            return False

        # Create numpy arrays
        points_L_img = np.array(points_L_img)  # POINTS_L
        points_R_img = np.array(points_R_img)  # POINTS_R
        points_L_img_ids = np.array(points_L_img_ids)  # ID_L
        points_R_img_ids = np.array(points_R_img_ids)  # ID_R

        pts_L_fund = np.int32(points_L_img)  # Transform float to int32
        pts_R_fund = np.int32(points_R_img)  # Transform float to int32

        F, mask = cv.findFundamentalMat(pts_L_fund, pts_R_fund)  # Find fundamental matrix using RANSARC
        # We select only inlier points
        pts_inlier_L = points_L_img[mask.ravel() == 1]  # Select inliers from imgL using fundamental mask
        pts_inlier_R = points_R_img[mask.ravel() == 1]  # Select inliers from imgR using fundamental mask
        # Select inlier IDS from imgL_index using fundamental mask
        pts_inlier_L_ids = points_L_img_ids[mask.ravel() == 1]
        # Select inlier IDS from imgR_index using fundamental mask
        pts_inlier_R_ids = points_R_img_ids[mask.ravel() == 1]

        pts_L_fund = pts_L_fund[mask.ravel() == 1]  # Select inliers from pts_L_funds for color finding
        pts_R_fund = pts_R_fund[mask.ravel() == 1]  # Select inliers from pts_R_funds for color finding

        color_inlier_L = find_color_list(self.imgL, pts_L_fund)  # find the corresponding color on left image
        color_inlier_R = find_color_list(self.imgR, pts_R_fund)  # find the corresponding color on right image

        color_inlier_L = np.array(color_inlier_L, dtype=np.int)
        color_inlier_R = np.array(color_inlier_R, dtype=np.int)

        self.imgL.img_set_features(pts_inlier_L, color_inlier_L, pts_inlier_L_ids)
        self.imgR.img_set_features(pts_inlier_R, color_inlier_R, pts_inlier_R_ids)

        g_inlier_size = len(pts_inlier_L)
        message_print("Found %d good inlier matches out of " % g_inlier_size + " %d good matches." % g_points_size)
        if g_inlier_size < 0.3 * g_points_size:
            message_print("Not a good match!")
            return False
        return self.m_img_create_model()

    def m_img_create_model(self):
        img_L_pnts = self.imgL.FEATURE_POINTS()  # Take the points of left image
        img_R_pnts = self.imgR.FEATURE_POINTS()  # Take the points of right image
        img_L_colors = self.imgL.FEATURE_COLORS()  # Take the color for points of left image
        img_R_colors = self.imgR.FEATURE_COLORS()  # Take the color for points of right image
        img_L_pnts_ids = self.imgL.FEATURE_IDS()  # Take the ids for points of left image
        img_R_pnts_ids = self.imgR.FEATURE_IDS()  # Take the ids for points of right image
        camera_matrix = self.imgL.CAMERA_MATRIX()  # Take the camera matrix of the left image (same camera matrix)

        E, mask = cv.findEssentialMat(img_L_pnts, img_R_pnts, camera_matrix)  # find essential matrix
        g_inlier_size = len(img_L_pnts)  # Take the size of img_L_pnts (same same with all lists)
        img_L_pnts = img_L_pnts[mask.ravel() == 1]  # Mask img_L_pnts
        img_R_pnts = img_R_pnts[mask.ravel() == 1]  # Mask img_R_pnts
        img_L_colors = img_L_colors[mask.ravel() == 1]  # Mask img_L_colors
        img_R_colors = img_R_colors[mask.ravel() == 1]  # Mask img_R_colors
        img_L_pnts_ids = img_L_pnts_ids[mask.ravel() == 1]  # Mask img_L_pnts_ids
        img_R_pnts_ids = img_R_pnts_ids[mask.ravel() == 1]  # Mask img_L_pnts_ids
        suggested_pair_model_size = len(img_L_pnts)  # Take the size of new img_L_pnts (same same with all lists)
        # Console Message
        message_print("Found %d suggested pair model points out of " % suggested_pair_model_size
                      + " %d good inlier matches." % g_inlier_size)

        retval, R, t, mask = cv.recoverPose(E, img_L_pnts, img_R_pnts, camera_matrix)  # Recover Pose

        # Console Message
        message_print("Found %d pair model points out of " % retval
                      + " %d suggested pair model points." % suggested_pair_model_size)
        
        if retval < 0.5*suggested_pair_model_size:
            message_print("Not enough points to create a good model.")
            return False

        message_print("Triangulate Pair Model")
        img_L_pnts = img_L_pnts[mask.ravel() == 255]  # Mask img_L_pnts
        img_R_pnts = img_R_pnts[mask.ravel() == 255]  # Mask img_R_pnts
        img_L_colors = img_L_colors[mask.ravel() == 255]  # Mask img_L_colors
        img_R_colors = img_R_colors[mask.ravel() == 255]  # Mask img_R_colors
        img_L_pnts_ids = img_L_pnts_ids[mask.ravel() == 255]  # Mask img_L_pnts_ids
        img_R_pnts_ids = img_R_pnts_ids[mask.ravel() == 255]  # Mask img_L_pnts_ids

        pose_matrix_L = pose_matrix_default()
        projection_matrix_L = projection_matrix_from_camera(camera_matrix)

        pose_matrix_R = pose_matrix_using_R_t(R, t)
        pose_matrix_R = pose_matrix_retransform_using_pair(pose_matrix_R, pose_matrix_L)
        R, t = pose_matrix_take_R_t(pose_matrix_R)
        projection_matrix_R = projection_matrix_from_pose_and_camera(R, t, self.imgR.camera.CAMERA_MATRIX())

        img_L_pnts = np.transpose(img_L_pnts)  # Set triangulation points for Left image
        img_R_pnts = np.transpose(img_R_pnts)  # Set triangulation points for Left image

        # Triangulate points
        points4D = cv.triangulatePoints(projMatr1=projection_matrix_L,
                                        projMatr2=projection_matrix_R,
                                        projPoints1=img_L_pnts,
                                        projPoints2=img_R_pnts)

        X_o = 0
        Y_o = 0
        Z_o = 0
        for i in range(0, retval):
            p_x = points4D[0][i] / points4D[3][i]
            p_y = points4D[1][i] / points4D[3][i]
            p_z = points4D[2][i] / points4D[3][i]
            p_tmp = [p_x, p_y, p_z]

            c_r = (img_L_colors[i][0] + img_R_colors[i][0]) / 2
            c_g = (img_L_colors[i][1] + img_R_colors[i][1]) / 2
            c_b = (img_L_colors[i][2] + img_R_colors[i][2]) / 2
            color_tmp = [c_r, c_g, c_b]

            id_tmp = [img_L_pnts_ids[i], img_R_pnts_ids[i]]

            X_o += p_x
            Y_o += p_y
            Z_o += p_z
            self.model_coord_list.append(p_tmp)
            self.model_color_list.append(color_tmp)
            self.model_coord_id_list.append(id_tmp)

        X_o /= retval
        Y_o /= retval
        Z_o /= retval

        # Move pair model to new principal point
        dx = 1000.0 - X_o
        dy = 1000.0 - Y_o
        dz = 1000.0 - Z_o

        for i in range(0, retval):
            self.model_coord_list[i][0] += dx
            self.model_coord_list[i][1] += dy
            self.model_coord_list[i][2] += dz

        # -------------------------------------------- #
        # Uncomment the following lines for debugging. #
        # -------------------------------------------- #
        # print(self.model_coord_list[0])
        # print(self.model_color_list[0])
        # print(self.model_coord_id_list[0])
        export_path = os.path.expanduser("~/Desktop")
        export_path += "/sfm_tmp"
        export_path_norm = os.path.normpath(export_path)
        if not os.path.exists(export_path_norm):
            os.mkdir(export_path_norm)
        export_path += "/" + self.imgL.IMG_NAME() + "_" + self.imgR.IMG_NAME() + ".ply"
        export_path_norm = os.path.normpath(export_path)
        export_as_ply(np.array(self.model_coord_list), np.array(self.model_color_list), export_path_norm)
        # -------------------------------------------- #

        return True


def pose_matrix_default():
    T = [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    return np.array(T)


def pose_matrix_using_R_t(R, t):
    Rt = []
    Rt.append(R)
    Rt.append(t)
    Rt = np.concatenate(Rt, axis=1)

    poseMtrx = []
    poseMtrx.append(Rt)
    poseMtrx.append([[0.0, 0.0, 0.0, 1.0]])
    poseMtrx = np.concatenate(poseMtrx, axis=0)
    return np.array(poseMtrx)


def pose_matrix_retransform_using_pair(base_matrix, transformational_matrix):
    p_mtrx = np.dot(transformational_matrix, base_matrix)
    return np.array(p_mtrx)


def pose_matrix_take_R_t(pose_matrix):
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3:]
    return np.array(R), np.array(t)


def projection_matrix_from_camera(cam_mtrx):
    projectionMtrx = []
    zeroMtrx = [[0], [0], [0]]
    projectionMtrx.append(cam_mtrx)
    projectionMtrx.append(zeroMtrx)
    projectionMtrx = np.concatenate(projectionMtrx, axis=1)
    return np.array(projectionMtrx)


def projection_matrix_from_pose_and_camera(R, t, cam_mtrx):
    R_t = np.transpose(R)
    m_R_t_t = np.dot(-R_t, t)

    P_tmp = []
    P_tmp.append(R_t)
    P_tmp.append(m_R_t_t)
    P_tmp = np.concatenate(P_tmp, axis=1)
    # print(P_tmp)

    P = np.dot(cam_mtrx, P_tmp)
    # print(P)
    return np.array(P)
