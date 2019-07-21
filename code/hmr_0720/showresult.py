import numpy as np
import cv2
import os


def show_result():
    while True:
        if os.path.isfile('/media/ramdisk/render_data/1/show/img_out_1.npy') :
	    result = np.load('/media/ramdisk/render_data/1/show/img_out_1.npy').reshape(720, 1920, 3)
            os.system("rm /media/ramdisk/render_data/1/show/img_out_1.npy")
        elif os.path.isfile('/media/ramdisk/render_data/2/show/img_out_2.npy') :
	    result = np.load('/media/ramdisk/render_data/2/show/img_out_2.npy').reshape(720, 1920, 3)
            os.system("rm /media/ramdisk/render_data/2/show/img_out_2.npy")
        elif os.path.isfile('/media/ramdisk/render_data/3/show/img_out_3.npy') :
	    result = np.load('/media/ramdisk/render_data/3/show/img_out_3.npy').reshape(720, 1920, 3)
            os.system("rm /media/ramdisk/render_data/3/show/img_out_3.npy")
        elif os.path.isfile('/media/ramdisk/render_data/4/show/img_out_4.npy') :
	    result = np.load('/media/ramdisk/render_data/4/show/img_out_4.npy').reshape(720, 1920, 3)
            os.system("rm /media/ramdisk/render_data/4/show/img_out_4.npy")
        else : continue

        cv2.imshow('Real Time Human 3d Rec', result)

        key = cv2.waitKey(25)
        if key == ord('q'):
            break
