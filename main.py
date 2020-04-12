from lib.sfm import *

path = "C:/Users/kavou/Desktop/data/video_frames/Talud_A2_01/"  # Give a path

sfm = SFM()  # Create SFM() object
# Run SFM
sfm.sfm_run(path, fp_method=FM_AKAZE, set_camera_method=CAM_APPROXIMATE)
