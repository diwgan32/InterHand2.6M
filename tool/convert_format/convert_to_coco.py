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

def reproject_to_3d(im_coords, K, z):
  im_coords = np.stack([im_coords[:,0], im_coords[:,1]],axis=1)
  im_coords = np.hstack((im_coords, np.ones((im_coords.shape[0],1))))
  projected = np.dot(np.linalg.inv(K), im_coords.T).T
  projected[:, 0] *= z
  projected[:, 1] *= z
  projected[:, 2] *= z
  return projected

def get_center(gtIn):
  wrist_1 = gtIn[0, 0]
  y = gtIn[0, 1]
  return [x, y]

def make_dirs(path):
  try:
    base = os.path.dirname(path)
    os.makedirs(base)
  except Exception as e:
    pass

def crop_and_center(imgInOrg, gtIn):
  shape = imgInOrg.shape
  box_size = min(imgInOrg.shape[0], imgInOrg.shape[1])
  center = get_center(gtIn)
  print(center, box_size)
  x_min_v = center[0] - box_size/2
  y_min_v = center[1] - box_size/2
  x_max_v = center[0] + box_size/2
  y_max_v = center[1] + box_size/2
  
  x_min_n = int(max(0, -x_min_v))
  y_min_n = int(max(0, -y_min_v))

  x_min_o = int(max(0, x_min_v))
  y_min_o = int(max(0, y_min_v))
  x_max_o = int(min(imgInOrg.shape[1], x_max_v))
  y_max_o = int(min(imgInOrg.shape[0], y_max_v))

  w = int(x_max_o - x_min_o)
  h = int(y_max_o - y_min_o)
  
  new_img = np.zeros((box_size, box_size, 3))
  new_img[y_min_n:y_min_n+h, x_min_n:x_min_n+w] = \
          imgInOrg[y_min_o:y_max_o, x_min_o:x_max_o]
  new_img = cv2.resize(new_img, (256, 256))
  x_min_v *= float(256)/480
  y_min_v *= float(256)/480
  return new_img, x_min_v, y_min_v

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

joint_num = 21
orig_root_idx = {'right': 20, 'left': 41}
orig_joint_type = {'right': np.arange(0,joint_num), 'left': np.arange(joint_num, joint_num*2)}

print(f"Num {split}: {len(db.anns.keys())}")

for aid in db.anns.keys():
    ann = db.anns[aid]
    image_id = ann['image_id']
    img = db.loadImgs(image_id)[0]
    capture_idx = img['capture']
    seq_name = img['seq_name']
    cam_idx = img['camera']
    frame_idx = img['frame_idx']

    img_path = osp.join(img_root_path, split, img['file_name'])
    joint_param = joints[capture_idx][frame_idx]
    campos, camrot = np.array(cam_params[str(capture_idx)]['campos'][str(cam_idx)], dtype=np.float32), np.array(cam_params[str(capture_idx)]['camrot'][str(cam_idx)], dtype=np.float32)
    focal, princpt = np.array(cam_params[str(capture_idx)]['focal'][str(cam_idx)], dtype=np.float32), np.array(cam_params[str(capture_idx)]['princpt'][str(cam_idx)], dtype=np.float32)

    joint_world = np.array(joints[str(capture_idx)][str(frame_idx)]['world_coord'], dtype=np.float32)
    joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
    joint_2d = cam2pixel(joint_cam, focal, princpt)[:,:2]
        
    img = cv2.imread(img_path)

    joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(self.joint_num*2)
    # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
    joint_valid[orig_joint_type['right']] *= joint_valid[orig_root_idx['right']]
    joint_valid[orig_joint_type['left']] *= joint_valid[orig_root_idx['left']]
            

    # processed_img, x_offset, y_offset = crop_and_center(img, joint_2d)
    # joint_2d[:, 0] -= x_offset
    # joint_2d[:, 1] -= y_offset
    # output_path = sample["color_file"].replace("DexYCB", "DexYCBOutput")
    # make_dirs(output_path)
    # cv2.imwrite(output_path, processed_img)
    # K = np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1.0]])
    # joint_3d = reproject_to_3d(joint_2d, K, joint_cam[:, 2])

    # joint_2d_aug, joint_3d_aug, joint_valid = left_or_right(joint_2d, joint_3d, sample["mano_side"])

    # output["images"].append({
    #   "id": count,
    #   "width": 256,
    #   "height": 256,
    #   "file_name": output_path,
    #   "camera_param": {
    #       "focal": [float(K[0][0]), float(K[1][1])],
    #       "princpt": [float(K[0][2]), float(K[1][2])]
    #   }
    # })

    # output["annotations"].append({
    #   "id": count,
    #   "image_id": count,
    #   "category_id": 1,
    #   "is_crowd": 0,
    #   "joint_img": joint_2d_aug.tolist(),
    #   "joint_valid": joint_valid,
    #   "hand_type": sample["mano_side"],
    #   "joint_cam": (joint_3d_aug * 1000).tolist(),
    #   "bbox": get_bbox(joint_2d)
    # })

    print(f"{ann['joint_valid']}, {joint_valid}, {joint_cam}")
    input("? ")
