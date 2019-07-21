"""
Script to convert openpose output into bbox
"""
import json
import numpy as np


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
        return -1, -1
    else: 
        print ("Json load success!")
    
    # Pick the most confident detection.
    if len(kps) == 0: return -1, -1
    print(kps)
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2] 
    if len(vis_kp) == 0: return -1, -1
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        
        print('bad!')
        return -1, -1
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    
    scale = 150. / person_height
    return scale, center



