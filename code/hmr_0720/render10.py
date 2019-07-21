import sys
import cv2
import os
import time
import json
import subprocess
from sys import platform
from absl import flags
import numpy as np
import skimage.io as io
import tensorflow as tf
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
import src.config
from src.RunModel import RunModel
from smpl_webuser.serialization import load_model
dir_path = os.path.dirname(os.path.realpath(__file__))
import multiprocessing


background_img = cv2.imread("/media/ramdisk/960720.png")


def render_1():
    renderer = vis_util.SMPLRenderer(face_path ='/home/feng/hmr_tf13/src/tf_smpl/smpl_faces.npy')
    render_count = 0
    render_time = time.time()
    while True:
        exists_1 = os.path.isfile('/media/ramdisk/render_data/1/vert_shifted.npy')
        exists_2 = os.path.isfile('/media/ramdisk/render_data/1/cam_for_render.npy')
        exists_3 = os.path.isfile('/media/ramdisk/render_data/1/kp.npy')
        if (exists_1 and exists_2 and exists_3) :
	    vert_shifted = np.load('/media/ramdisk/render_data/1/vert_shifted.npy')
	    cam_for_render = np.load('/media/ramdisk/render_data/1/cam_for_render.npy')
	    kp = np.load('/media/ramdisk/render_data/1/kp.npy')
            if((vert_shifted.size == 6890*3) and (cam_for_render.size == 3) and (kp.size == 720*960*3)) :
                vert_shifted.reshape(6890, 3)
		kp.reshape(720, 960, 3)
                os.system("rm /media/ramdisk/render_data/1/*.npy")
            else : continue
        else : 
            #cv2.waitkey(5)
            continue

	rend_img = renderer(vert_shifted, cam_for_render, img_size=[720, 960], img=background_img)
	if kp is None : continue
        img_out_1 = np.hstack((kp, rend_img))
	np.save('/media/ramdisk/render_data/1/img_out_1.npy',img_out_1)
        os.system("mv /media/ramdisk/render_data/1/img_out_1.npy /media/ramdisk/render_data/1/show/")
        #cv2.imwrite('/media/ramdisk/render_data/1/result.jpg', img_out_1)
        render_count = render_count + 1
        if render_count == 100 :
            print('render 1 FPS:', 1.0 / ((time.time() - render_time)/ 100.0))
            render_count = 0
      	    render_time = time.time() 



def render_2():
    renderer = vis_util.SMPLRenderer(face_path ='/home/feng/hmr_tf13/src/tf_smpl/smpl_faces.npy')
    render_count = 0
    render_time = time.time()
    while True:
        exists_1 = os.path.isfile('/media/ramdisk/render_data/2/vert_shifted.npy')
        exists_2 = os.path.isfile('/media/ramdisk/render_data/2/cam_for_render.npy')
        exists_3 = os.path.isfile('/media/ramdisk/render_data/2/kp.npy')
        if (exists_1 and exists_2 and exists_3) :
	    vert_shifted = np.load('/media/ramdisk/render_data/2/vert_shifted.npy')
	    cam_for_render = np.load('/media/ramdisk/render_data/2/cam_for_render.npy')
	    kp = np.load('/media/ramdisk/render_data/2/kp.npy')
            if((vert_shifted.size == 6890*3) and (cam_for_render.size == 3) and (kp.size == 720*960*3)) :
                vert_shifted.reshape(6890,3)
		kp.reshape(720, 960, 3)
                os.system("rm /media/ramdisk/render_data/2/*.npy")
            else : continue
        else : 
            #cv2.waitkey(5)
            continue

	rend_img = renderer(vert_shifted, cam_for_render, img_size=[720, 960], img=background_img)
	if kp is None : continue
	img_out_2 = np.hstack((kp, rend_img))
	np.save('/media/ramdisk/render_data/2/img_out_2.npy',img_out_2)
        os.system("mv /media/ramdisk/render_data/2/img_out_2.npy /media/ramdisk/render_data/2/show/")
        render_count = render_count + 1
        if render_count == 100 :
            print('render 2 FPS:', 1.0 / ((time.time() - render_time)/ 100.0))
            render_count = 0
      	    render_time = time.time() 

def render_3():
    renderer = vis_util.SMPLRenderer(face_path ='/home/feng/hmr_tf13/src/tf_smpl/smpl_faces.npy')
    render_count = 0
    render_time = time.time()
    while True:
        exists_1 = os.path.isfile('/media/ramdisk/render_data/3/vert_shifted.npy')
        exists_2 = os.path.isfile('/media/ramdisk/render_data/3/cam_for_render.npy')
        exists_3 = os.path.isfile('/media/ramdisk/render_data/3/kp.npy')
        if (exists_1 and exists_2 and exists_3) :
	    vert_shifted = np.load('/media/ramdisk/render_data/3/vert_shifted.npy')
	    cam_for_render = np.load('/media/ramdisk/render_data/3/cam_for_render.npy')
	    kp = np.load('/media/ramdisk/render_data/3/kp.npy')
            if((vert_shifted.size == 6890*3) and (cam_for_render.size == 3) and (kp.size == 720*960*3)) :
                vert_shifted.reshape(6890,3)
		kp.reshape(720, 960, 3)
                os.system("rm /media/ramdisk/render_data/3/*.npy")
            else : continue
        else : 
            #cv2.waitkey(5)
            continue

	rend_img = renderer(vert_shifted, cam_for_render, img_size=[720, 960], img=background_img)
	if kp is None : continue
	img_out_3 = np.hstack((kp, rend_img))
	np.save('/media/ramdisk/render_data/3/img_out_3.npy',img_out_3)
        os.system("mv /media/ramdisk/render_data/3/img_out_3.npy /media/ramdisk/render_data/3/show/")
        render_count = render_count + 1
        if render_count == 100 :
            print('render 3 FPS:', 1.0 / ((time.time() - render_time)/ 100.0))
            render_count = 0
      	    render_time = time.time() 




def render_4():
    renderer = vis_util.SMPLRenderer(face_path ='/home/feng/hmr_tf13/src/tf_smpl/smpl_faces.npy')
    render_count = 0
    render_time = time.time()
    while True:
        exists_1 = os.path.isfile('/media/ramdisk/render_data/4/vert_shifted.npy')
        exists_2 = os.path.isfile('/media/ramdisk/render_data/4/cam_for_render.npy')
        exists_3 = os.path.isfile('/media/ramdisk/render_data/4/kp.npy')
        if (exists_1 and exists_2 and exists_3) :
	    vert_shifted = np.load('/media/ramdisk/render_data/4/vert_shifted.npy')
	    cam_for_render = np.load('/media/ramdisk/render_data/4/cam_for_render.npy')
	    kp = np.load('/media/ramdisk/render_data/4/kp.npy')
            if((vert_shifted.size == 6890*3) and (cam_for_render.size == 3) and (kp.size == 720*960*3)) :
                vert_shifted.reshape(6890,3)
		kp.reshape(720, 960, 3)
                os.system("rm /media/ramdisk/render_data/4/*.npy")
            else : continue
        else : 
            #cv2.waitkey(5)
            continue

	rend_img = renderer(vert_shifted, cam_for_render, img_size=[720, 960], img=background_img)
	if kp is None : continue
	img_out_4 = np.hstack((kp, rend_img))
        cv2.imwrite("1111.jpg",img_out_4)
	#np.save('/media/ramdisk/render_data/4/img_out_4.npy',img_out_4)
        os.system("mv /media/ramdisk/render_data/4/img_out_4.npy /media/ramdisk/render_data/4/show/")
        render_count = render_count + 1
        if render_count == 100 :
            print('render 4 FPS:', 1.0 / ((time.time() - render_time)/ 100.0))
            render_count = 0
      	    render_time = time.time() 



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

        key = cv2.waitKey(1)
        if key == ord('q'):
            p1.terminate() 
            p2.terminate() 
            p3.terminate() 
            p4.terminate()  

pipe_1=multiprocessing.Pipe()
pipe_2=multiprocessing.Pipe()
pipe_3=multiprocessing.Pipe()
pipe_4=multiprocessing.Pipe()


p1 = multiprocessing.Process(target=render_1) 
p2 = multiprocessing.Process(target=render_2) 
p3 = multiprocessing.Process(target=render_3) 
p4 = multiprocessing.Process(target=render_4) 
#p5 = multiprocessing.Process(target=show_result) 



p1.start()
p2.start()
p3.start()
p4.start()
#p5.start()
#print(11111)
#show_result()

