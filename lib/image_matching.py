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

    # ---------------- #
    # RETURN FUNCTIONS #
    # ---------------- #
    def IMG_MATCH_ID(self):
        return self.m_img_id

    def IMG_LEFT(self):
        return self.imgL

    def IMG_RIGHT(self):
        return self.imgR

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

        color_inlier_L = find_color_list(self.imgL, pts_L_fund)  # find the corresponding color on left image
        color_inlier_R = find_color_list(self.imgR, pts_R_fund)  # find the corresponding color on right image

        self.imgL.img_set_features(pts_inlier_L, color_inlier_L, pts_inlier_L_ids)
        self.imgR.img_set_features(pts_inlier_R, color_inlier_R, pts_inlier_R_ids)

        g_inlier_size = len(pts_inlier_L)
        message_print("Found %d good inlier matches out of " % g_inlier_size + " %d good matches." % g_points_size)
        if g_inlier_size < 0.3 * g_points_size:
            message_print("Not a good match!")
            return False
        return True
