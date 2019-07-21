from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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


flags.DEFINE_string('json_path', '/media/ramdisk/output_op/keypoints.json', 'Json file')
flags.DEFINE_string('img_path', '/media/ramdisk/output_op/input.jpg', 'Image to run')

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1
renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
sess = tf.Session()
model = RunModel(config, sess=sess)
smpl_model_used = load_model('/home/feng/Documents/zhangjiang/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
background_img = cv2.imread("/media/ramdisk/960720.png")
outmesh_path = '/media/ramdisk/result.obj'
data = {}


def write_obj(smpl_model_used, v, outmesh_path):
    
    smpl_model_used.pose[:] = v[0][3:75]
    smpl_model_used.betas[:] = v[0][75:85]
    with open( outmesh_path, 'w') as fp:
        for v in smpl_model_used.r:
            fp.write('v %f %f %f\n' % (v[0],v[1],v[2]))

        for f in smpl_model_used.f + 1:
            fp.write('f %d %d %d\n' %(f[0], f[1], f[2]))
    
    print('..Output mesh saved to: ', outmesh_path)

    return 1


def write_camera_pose(data):
    with open('/media/ramdisk/camera.json', 'w') as json_file:
        json_file.write(json.dumps(data))
        print('camera pose writed!')

while True:
    t0 = time.time()
    try:
        img = io.imread(config.img_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]
    except IOError:
        print("image not found, try again!")
        continue
    else: 
        print("image load success!")
    scale, center = op_util.get_bbox(config.json_path)
    if scale == -1 and center == -1: continue
    if scale >= 10: continue
    #print(111, scale, center, config.img_size)
    input_img, proc_param = img_util.scale_and_crop(img, scale, center, config.img_size)
    input_img = 2 * ((input_img / 255.) - 0.5)
    input_img = np.expand_dims(input_img, 0)
    joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
    #print('3D Rec:', time.time() - t0)
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(proc_param, verts[0], cams[0], joints[0], img_size=img.shape[:2])
    #print('3D Rec:', time.time() - t0)
    #print('type(cam_for_render):', type(cam_for_render))
    #print(img.shape[:2])    
    
    #write_obj(smpl_model_used,theta,outmesh_path)
    #str_camera_pose = ''.join(cam_for_render)
    #data["flenth"] = cam_for_render[0]
    #data["px"] = cam_for_render[1]
    #data["py"] = cam_for_render[2]
    #data["camera_pose"]=cam_for_render.tolist()
    #write_camera_pose(data)
    #exit()
    #open("/home/feng/Documents/zhangjiang/rtviewer/public/resource/model/result1.obj", "wb").write(
	#open(outmesh_path, "rb").read())
    #os.system("mv /home/feng/Documents/zhangjiang/rtviewer/public/resource/model/result1.obj /home/feng/Documents/zhangjiang/rtviewer/public/resource/model/result.obj")
    #os.system("mv /media/ramdisk/result.obj /media/ramdisk/hmr_json/result.obj")
    rend_img = renderer(vert_shifted, cam_for_render, img_size=[720, 960], img=background_img)

    #print('3D Rec:', time.time() - t0)

    if rend_img is None: continue
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #imgzi = cv2.putText(rend_img, "SC: {sc:.2f}, F: {fl:.2f}, PX: {px:.2f}, PY: {py:.2f}".format(sc=scale, fl=cam_for_render[0],px=cam_for_render[1], py=cam_for_render[2]), (50, 400), font, 0.6, (0, 0, 255), 2)    

    #imgzi = cv2.putText(imgzi, "SC: {sc:.2f}, F: {fl:.2f}, PX: {px:.2f}, PY: {py:.2f}".format(sc=scale, fl=cam_for_render[0]*scale,px=cam_for_render[1]*scale, py=cam_for_render[2]*scale), (50, 430), font, 0.6, (0, 0, 255), 2)   

    cv2.imshow("rend_img", rend_img)
    print('3D Rec:', time.time() - t0)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 

#cv2.destroyAllWindows()










