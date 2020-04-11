from lib.sfm import *

path = "C:/Users/kavou/Desktop/data/video_frames/Talud_A2_01/"

sfm = SFM()
sfm.sfm_run(path, fp_method=FM_AKAZE)
