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
sys.path.append('/home/feng/Documents/zhangjiang/SMPL_python_v.1.0.0/smpl/')
from smpl_webuser.serialization import load_model
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/home/feng/openpose_py27/build/python');
import threading 
import glob
import multiprocessing



try:

    from openpose import pyopenpose as op
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')


def set_params():
    params = dict()
    params["num_gpu"] = 1
    params["num_gpu_start"] = 1
    params["net_resolution"] = "320x176"
    params["model_pose"] = "BODY_25"
    params["model_folder"] = "/home/feng/openpose/models/"
    params["write_json"] = "/media/ramdisk/output_op/"
    return params


def capture_image(pipe_img):
    
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    capture_count = 0
    capture_time = time.time()
    while True:    
        ret,img = stream.read()
        if img is None : continue
        img = img[:,160:1120,:]
        pipe_img.send(img)
        capture_count = capture_count + 1
        if capture_count == 300 :
            print('Capture FPS:', 1.0 / ((time.time() - capture_time)/ 300.0))
            capture_count = 0
      	    capture_time = time.time() 
    return 0


def detection(pipe_img,pipe_center,pipe_scale,pipe_img_2,pipe_kp):

    params = set_params()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    detection_count = 0
    detection_time = time.time()
    while True: 
        img = pipe_img.recv()
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop([datum])
        bodyKeypoints_img = datum.cvOutputData
        cv2.rectangle(bodyKeypoints_img,(330,620),(630,720),(0,0,255),3)
        #cv2.imwrite('kps.jpg',bodyKeypoints_img)
        json_path = glob.glob('/media/ramdisk/output_op/*keypoints.json')
        scale, center = op_util.get_bbox(json_path[0])
        if scale == -1 and center == -1: continue
        if scale >= 10: continue
        pipe_img_2.send(img)
        pipe_center.send(center)
        pipe_scale.send(scale)
        pipe_kp.send(bodyKeypoints_img)
        os.system("rm /media/ramdisk/output_op/*keypoints.json")
        detection_count = detection_count + 1
        if detection_count == 100 :
            print('Detection FPS:', 1.0 / ((time.time() - detection_time)/ 100.0))
            detection_count = 0
      	    detection_time = time.time() 


def rec_human(pipe_img_2,pipe_center,pipe_scale,pipe_kp):
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    rec_human_count = 0
    rec_human_time = time.time()
    num_render = 1

    while True:

        img = pipe_img_2.recv()
        center = pipe_center.recv()
        scale = pipe_scale.recv()
        kp = pipe_kp.recv()
        input_img, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)
        input_img = 2 * ((input_img / 255.) - 0.5) 
        input_img = np.expand_dims(  input_img, 0)
        joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(proc_param, verts[0], cams[0], joints[0], img_size=img.shape[:2])

        print(111111)
        if num_render == 1:
            np.save('/media/ramdisk/render_data/1/cam_for_render.npy',cam_for_render)
            np.save('/media/ramdisk/render_data/1/vert_shifted.npy',vert_shifted)
	    np.save('/media/ramdisk/render_data/1/kp.npy',kp)
            #cv2.imwrite('/media/ramdisk/render_data/1/kp.jpg',kp)
            #print(kp.shape)
            num_render = 2
        if num_render == 2:
            np.save('/media/ramdisk/render_data/2/cam_for_render.npy',cam_for_render)
            np.save('/media/ramdisk/render_data/2/vert_shifted.npy',vert_shifted)
	    np.save('/media/ramdisk/render_data/2/kp.npy',kp)
	    #cv2.imwrite('/media/ramdisk/render_data/2/kp.jpg',kp)
            num_render = 3
        if num_render == 3:
            np.save('/media/ramdisk/render_data/3/cam_for_render.npy',cam_for_render)
            np.save('/media/ramdisk/render_data/3/vert_shifted.npy',vert_shifted)
	    np.save('/media/ramdisk/render_data/3/kp.npy',kp)
	    #cv2.imwrite('/media/ramdisk/render_data/3/kp.jpg',kp)
            num_render = 4
        if num_render == 4:
            np.save('/media/ramdisk/render_data/4/cam_for_render.npy',cam_for_render)
            np.save('/media/ramdisk/render_data/4/vert_shifted.npy',vert_shifted)
	    np.save('/media/ramdisk/render_data/4/kp.npy',kp)
	    #cv2.imwrite('/media/ramdisk/render_data/4/kp.jpg',kp)
            num_render = 1

        rec_human_count = rec_human_count + 1
        if rec_human_count == 100 :
            print('rec FPS:', 1.0 / ((time.time() - rec_human_time)/ 100.0))
            rec_human_count = 0
      	    rec_human_time = time.time()


pipe_img=multiprocessing.Pipe()
pipe_img_2=multiprocessing.Pipe()
pipe_center=multiprocessing.Pipe()
pipe_scale=multiprocessing.Pipe()
pipe_kp = multiprocessing.Pipe()


p1 = multiprocessing.Process(target=capture_image, args=(pipe_img[0],)) 
p2 = multiprocessing.Process(target=detection, args=(pipe_img[1],pipe_center[0],pipe_scale[0],pipe_img_2[0],pipe_kp[0],)) 
p3 = multiprocessing.Process(target=rec_human, args=(pipe_img_2[1],pipe_center[1],pipe_scale[1],pipe_kp[1],)) 


p1.start()
p2.start()
p3.start()





