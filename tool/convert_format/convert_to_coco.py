import os
import numpy as np
import cv2
import json
from glob import glob
import os.path as osp
import trimesh
import sys
sys.path.insert(0, osp.join('../../', 'common'))
sys.path.insert(0, osp.join('../../', 'main'))
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.preprocessing import load_skeleton
from pycocotools.coco import COCO

def vis_keypoints(img, kps, skeleton):
    for i in range(21):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))
        #print("Score", score[i], score[pid], pid)
        img = cv2.line(img, (int(kps[i][0]), int(kps[i][1])), (int(kps[pid][0]), int(kps[pid][1])), color=(0, 0, 0), thickness=1)

    return img
          
root_path = '/home/ubuntu/RawDatasets/InterHand/InterHand2.6M_5fps_batch1'
img_root_path = osp.join(root_path, 'images')
annot_root_path = osp.join(root_path, 'annotations')
split = "train"

skeleton = load_skeleton('../../data/InterHand2.6M/annotations/skeleton.txt', 42)

db = COCO(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_data.json'))
with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_camera.json')) as f:
    cam_params = json.load(f)
with open(osp.join(annot_root_path, split, 'InterHand2.6M_' + split + '_joint_3d.json')) as f:
    joints = json.load(f)


print(f"Num {split}: {len(db.anns.keys())}")

for aid in db.anns.keys():
    ann = db.anns[aid]
    image_id = ann['image_id']
    img = db.loadImgs(image_id)[0]
    capture_id = img['capture']
    seq_name = img['seq_name']
    cam = img['camera']
    frame_idx = img['frame_idx']
    print(f"{image_id}, {capture_id}, {seq_name}, {cam}, {frame_idx}")
    input("? ")
