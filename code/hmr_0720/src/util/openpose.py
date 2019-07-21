"""
Script to convert openpose output into bbox
"""
import json
import numpy as np
from rsmq import RedisSMQ
import base64
import json

QUEUE3 = "smplOrigQue";
queue3 = RedisSMQ(host="127.0.0.1", qname=QUEUE3)
msg3 = []
kp_human = {}

def write_human_pose(data):
    with open('/home/feng/human_zj/rtviewer/public/resource/model/human.json', 'w') as json_file:
        json_file.write(json.dumps(data))
        #str_3 = json_file.read()

        #print('human pose writed!')


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps


def get_bbox(json_path, vis_thr=0.2):
    

    try:
        kps = read_json(json_path)
    except IOError:
        print ("Json not found, try again!")
        return -1, -1, -1

    if len(kps) == 0: return -1, -1, -1
    
    todelete = []
    for i in range(0, len(kps)):
	if i > 0:
#        if  ((kps[i][22,1] < 630 or kps[i][22,0] < 330 or kps[i][22,0] > 630) and (kps[i][19,1] < 630 or kps[i][19,0] < 330 or kps[i][19,0] > 630)):
	    todelete.append(i)
    for i in reversed(todelete):
	del kps[i]

    if len(kps) == 0: return -1, -1, -1

 


    scores_height = [kp[:,1].max()-kp[:,1].min() for kp in kps]
    kp = kps[np.argmax(scores_height)]
    
    #print(kp.shape)


    kp_human["Nose_x"] = kp[0,0]
    kp_human["Nose_y"] = kp[0,1]

    kp_human["Neck_x"] = kp[1,0]
    kp_human["Neck_y"] = kp[1,1]

    kp_human["RShoulder_x"] = kp[2,0]
    kp_human["RShoulder_y"] = kp[2,1]

    kp_human["RElbow_x"] = kp[3,0]
    kp_human["RElbow_y"] = kp[3,1]

    kp_human["RWrist_x"] = kp[4,0]
    kp_human["RWrist_y"] = kp[4,1]

    kp_human["LShoulder_x"] = kp[5,0]
    kp_human["LShoulder_y"] = kp[5,1]

    kp_human["LElbow_x"] = kp[6,0]
    kp_human["LElbow_y"] = kp[6,1]

    kp_human["LWrist_x"] = kp[7,0]
    kp_human["LWrist_y"] = kp[7,1]

    kp_human["MidHip_x"] = kp[8,0]
    kp_human["MidHip_y"] = kp[8,1]

    kp_human["RHip_x"] = kp[9,0]
    kp_human["RHip_y"] = kp[9,1]

    kp_human["RKnee_x"] = kp[10,0]
    kp_human["RKnee_y"] = kp[10,1]

    kp_human["RAnkle_x"] = kp[11,0]
    kp_human["RAnkle_y"] = kp[11,1]

    kp_human["LHip_x"] = kp[12,0]
    kp_human["LHip_y"] = kp[12,1]

    kp_human["LKnee_x"] = kp[13,0]
    kp_human["LKnee_y"] = kp[13,1]

    kp_human["LAnkle_x"] = kp[14,0]
    kp_human["LAnkle_y"] = kp[14,1]

    kp_human["REye_x"] = kp[15,0]
    kp_human["REye_y"] = kp[15,1]

    kp_human["LEye_x"] = kp[16,0]
    kp_human["LEye_y"] = kp[16,1]

    kp_human["REar_x"] = kp[17,0]
    kp_human["REar_y"] = kp[17,1]

    kp_human["LEar_x"] = kp[18,0]
    kp_human["LEar_y"] = kp[18,1]

    kp_human["LBigToe_x"] = kp[19,0]
    kp_human["LBigToe_y"] = kp[19,1]

    kp_human["LSmallToe_x"] = kp[20,0]
    kp_human["LSmallToe_y"] = kp[20,1]

    kp_human["LHeel_x"] = kp[21,0]
    kp_human["LHeel_y"] = kp[21,1]

    kp_human["RBigToe_x"] = kp[22,0]
    kp_human["RBigToe_y"] = kp[22,1]

    kp_human["RSmallToe_x"] = kp[23,0]
    kp_human["RSmallToe_y"] = kp[23,1]

    kp_human["RHeel_x"] = kp[24,0]
    kp_human["RHeel_y"] = kp[24,1]

    write_human_pose(kp_human)
    #str_3 = open('/home/feng/Downloads/rtviewer/public/resource/model/human.json', 'rb').read()
    #message_id = queue3.sendMessage(delay=0).message(str_3).execute()
    #msg3.append(message_id)


    if len(msg3) > 1:
        rt = queue3.deleteMessage(id=msg3[0]).execute()
        del msg3[0]


    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2] 
    if len(vis_kp) == 0: return -1, -1, -1
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        
        print('bad!')
        return -1, -1, -1
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    
    scale = 150. / person_height

    person_shape = max_pt - min_pt
    return scale, center, person_shape



