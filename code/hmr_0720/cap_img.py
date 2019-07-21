import sys
import cv2
import os
import time
import json
import signal
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
from rsmq import RedisSMQ
import base64

# google-chrome http://localhost:3000 --start-fullscreen --incognito
smpl_model_used = load_model('/home/feng/Documents/zhangjiang/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
outmesh_path = '/media/ramdisk/result.obj'
QUEUE1 = "smplOrigQue";
QUEUE2 = "smplObjQue";


queue1 = RedisSMQ(host="127.0.0.1", qname=QUEUE1)
queue2 = RedisSMQ(host="127.0.0.1", qname=QUEUE2)

try:
    queue1.deleteQueue().exceptions(False).execute()
except Exception as e:
    print(e)    
    
try:
    queue1.createQueue(delay=0).vt(0).maxsize(-1).execute()
except Exception as e:
    print(e)

try:
    queue2.deleteQueue().exceptions(False).execute()
except Exception as e:
    print(e)    
    
try:
    queue2.createQueue(delay=0).vt(0).maxsize(-1).execute()
except Exception as e:
    print(e)

msg1 = []
msg2 = []

try:

    from openpose import pyopenpose as op
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

last_person = [0,0,0,0,0]

pose_list = [np.zeros(72), np.zeros(72), np.zeros(72)]

def set_params():
    params = dict()
    params["num_gpu"] = 1
    params["num_gpu_start"] = 1
    params["net_resolution"] = "320x176"
    #params["net_resolution"] = "640x352"
    params["model_pose"] = "BODY_25"
    params["model_folder"] = "/home/feng/openpose/models/"
    params["write_json"] = "/media/ramdisk/output_op/"
    return params


def capture_image(pipe_img):
    
    # stream = cv2.VideoCapture(-1)
    # stream.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    # stream.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    capture_count = 0
    capture_time = time.time()
    while True:    
        # ret,img = stream.read()/
        img0 = cv2.imread("./im3.jpg")
        # img1 = cv2.resize(img0, (500, 670))
        # # img = np.pad(
        # # img1, ((100, 0), (330, 330), (0, 0)), mode='edge')
        # img = np.pad(
        # img1, ((50, 0), (390, 390), (0, 0)), mode='edge')
        # if img is None : continue
        # img = img[:,160:1120,:]
        # img = cv2.flip(img,1,dst=None)
        pipe_img.send(img0)
        capture_count = capture_count + 1
        if capture_count == 300 :
            print('Capture FPS:', 1.0 / ((time.time() - capture_time)/ 300.0))
            capture_count = 0
      	    capture_time = time.time() 
    return 0


def write_obj(smpl_model_used, v, outmesh_path):
    global pose_list
    print('%.1f' % (v[0][3]/np.pi*180)),
    print('%.1f' % (v[0][4]/np.pi*180)),
    print('%.1f' % (v[0][5]/np.pi*180))

    v[0][55] = 0.8

    print(v[0][57],v[0][58],v[0][59])
    v[0][57] = 0.8

    v[0][60] = 0
    v[0][61] = 0
    v[0][62] = 0

    tmp_pose = v[0][3:75]

    # pose_list[0] = pose_list[1]
    # pose_list[1] = pose_list[2]
    # pose_list[2] = tmp_pose
    # # if (max(pose_list[0]) == 0 and min(pose_list[0]) == 0) or (max(pose_list[1]) == 0 and min(pose_list[1]) == 0) or (max(pose_list[2]) == 0 and min(pose_list[2]) == 0):
    # # 	pose_list[0] = pose_list[1]
    # #     pose_list[1] = pose_list[2]
    # #     pose_list[2] = tmp_pose
    # #     return

    # for i in range(24):
    # 	tmp_pose[i*3] = np.mean([pose_list[0][i*3],pose_list[1][i*3],pose_list[2][i*3]])
    # 	tmp_pose[i*3+1] = np.mean([pose_list[0][i*3+1],pose_list[1][i*3+1],pose_list[2][i*3+1]])
    # 	tmp_pose[i*3+2] = np.mean([pose_list[0][i*3+2],pose_list[1][i*3+2],pose_list[2][i*3+2]])

    # pose_list[2] = tmp_pose
    smpl_model_used.pose[:] = tmp_pose
    smpl_model_used.betas[:] = v[0][75:85]

    # diff = np.abs(pose_list[2] - pose_list[1])
    # print(max(diff))

    # if max(diff) > 0.5:
    # 	return

    with open( outmesh_path, 'w') as fp:
        for v in smpl_model_used.r:
            fp.write('v %f %f %f\n' % (v[0],v[1],v[2]))

        for f in smpl_model_used.f + 1:
            fp.write('f %d %d %d\n' %(f[0], f[1], f[2]))
    
    #print('..Output mesh saved to: ', outmesh_path)

    return 1



def detection(pipe_img,pipe_center,pipe_scale,pipe_shape,pipe_img_2,pipe_kp):

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
        # cv2.rectangle(bodyKeypoints_img,(330,100),(630,720),(0,0,255),1)
        # cv2.rectangle(bodyKeypoints_img,(330,630),(630,720),(0,0,255),3)
        cv2.imwrite('/media/ramdisk/kps.jpg',bodyKeypoints_img)
        str_img_kps = base64.b64encode(open('/media/ramdisk/kps.jpg', 'rb').read())
        message_id = queue1.sendMessage(delay=0).message(str_img_kps.decode('utf-8')).execute()
        msg1.append(message_id)
        if len(msg1) > 1:
            rt = queue1.deleteMessage(id=msg1[0]).execute()
            del msg1[0]

        json_path = glob.glob('/media/ramdisk/output_op/*keypoints.json')
        scale, center, person_shape = op_util.get_bbox(json_path[0])
        if scale == -1 and center == -1 and person_shape == -1: continue
        if scale >= 10: continue
        pipe_img_2.send(img)
        pipe_center.send(center)
        pipe_scale.send(scale)
        pipe_shape.send(person_shape)
        pipe_kp.send(bodyKeypoints_img)
        os.system("rm /media/ramdisk/output_op/*keypoints.json")
        detection_count = detection_count + 1
        if detection_count == 100 :
            print('Detection FPS:', 1.0 / ((time.time() - detection_time)/ 100.0))
            detection_count = 0
      	    detection_time = time.time() 


def rec_human(pipe_img_2,pipe_center,pipe_scale,pipe_shape,pipe_kp):
    global last_person
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    rec_human_count = 0
    rec_human_time = time.time()
    #num_render = 1

    while True:

        img = pipe_img_2.recv()
        center = pipe_center.recv()
        scale = pipe_scale.recv()
        person_shape = pipe_shape.recv()
        kp = pipe_kp.recv()

        input_img, proc_param, last_person = img_util.scale_and_crop(img, scale, center, person_shape, 0.25, config.img_size, last_person)
        cv2.imwrite('/media/ramdisk/input.jpg',input_img)
        print(np.mean(input_img))

        input_img = ((input_img / 255.)) 

        # input_img = 2 * ((input_img / 255.) - 0.5) 

        input_img = np.expand_dims(  input_img, 0)
        joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
        #cam_for_render, vert_shifted, joints_orig = vis_util.get_original(proc_param, verts[0], cams[0], joints[0], img_size=img.shape[:2])
        write_obj(smpl_model_used,theta,outmesh_path)
        str_1 = open(outmesh_path, 'rb').read()
        message_id = queue2.sendMessage(delay=0).message(str_1).execute()
        msg2.append(message_id)
        if len(msg2) > 1:
            rt = queue2.deleteMessage(id=msg2[0]).execute()
            del msg2[0]


        rec_human_count = rec_human_count + 1
        if rec_human_count == 100 :
            print('rec FPS:', 1.0 / ((time.time() - rec_human_time)/ 100.0))
            rec_human_count = 0
      	    rec_human_time = time.time()


def sigintHandler(signum, frame):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

if __name__ == "__main__":

	signal.signal(signal.SIGINT, sigintHandler)
	signal.signal(signal.SIGHUP, sigintHandler)
	signal.signal(signal.SIGTERM, sigintHandler)

	pipe_img=multiprocessing.Pipe()
	pipe_img_2=multiprocessing.Pipe()
	pipe_center=multiprocessing.Pipe()
	pipe_scale=multiprocessing.Pipe()
	pipe_kp = multiprocessing.Pipe()
	pipe_shape = multiprocessing.Pipe()


	p1 = multiprocessing.Process(target=capture_image, args=(pipe_img[0],))
	p1.daemon = True
	p2 = multiprocessing.Process(target=detection, args=(pipe_img[1],pipe_center[0],pipe_scale[0],pipe_shape[0],pipe_img_2[0],pipe_kp[0],)) 
	p2.daemon = True
	p3 = multiprocessing.Process(target=rec_human, args=(pipe_img_2[1],pipe_center[1],pipe_scale[1],pipe_shape[1],pipe_kp[1],)) 
	p3.daemon = True

	p1.start()
	p2.start()
	p3.start()
	try:
		p1.join()
		p2.join()
		p3.join()
	except Exception as e:
		print str(e)





