from lib.sfm import *

#path = "C:/Users/kavou/Desktop/data/video_frames/Talud_A2_01/"  # Give a path
#path = "/home/johncrabs/Desktop/img_tmp/Talud_A2_01/"
path = "/home/johncrabs/Desktop/CalibData/RedMi_Test/"


sfm = SFM()  # Create SFM() object
# Run SFM
sfm.sfm_run(path, fp_method=FM_AKAZE, set_camera_method=CAM_APPROXIMATE, match_method=MATCH_FLANN,
            set_quality=Q_MEDIUM)

#sfm2 = SFM()
#sfm2.sfm_run(path, fp_method=FM_AKAZE, set_camera_method=CAM_APPROXIMATE,
#             match_method=MATCH_BRUTEFORCE_HAMMING)
