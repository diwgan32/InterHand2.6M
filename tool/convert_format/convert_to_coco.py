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

def get_center(gtIn, joint_valid):
    wrist_right = np.array([
        gtIn[orig_root_idx["right"], 0],
        gtIn[orig_root_idx["right"], 1]
    ])
    wrist_left = np.array([
        gtIn[orig_root_idx["left"], 0],
        gtIn[orig_root_idx["left"], 1]
    ])

    # Mask the left/right wrist when
    # computing center if its not valid
    
    wrist_list = []
    if (joint_valid[orig_root_idx['right']] == 1):
        wrist_list.append(wrist_right)
    if (joint_valid[orig_root_idx['left']] == 1):
        wrist_list.append(wrist_left)

    return np.average(wrist_list, axis=0)

def make_dirs(path):
    try:
        base = os.path.dirname(path)
        os.makedirs(base)
    except Exception as e:
        pass

def crop_and_center(imgInOrg, gtIn, joint_valid):
    shape = imgInOrg.shape
    box_size = min(imgInOrg.shape[0], imgInOrg.shape[1])
    center = get_center(gtIn, joint_valid)
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
    x_min_v *= float(256)/box_size
    y_min_v *= float(256)/box_size
    return new_img, x_min_v, y_min_v, float(256)/box_size

def vis_keypoints(img, kps, skeleton, bbox, joint_valid):
    for i in range(42):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        if (pid == -1):
            continue
        if (not joint_valid[i] or not joint_valid[pid]):
            continue
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))
        #print("Score", score[i], score[pid], pid)
        img = cv2.line(img, (int(kps[i][0]), int(kps[i][1])), (int(kps[pid][0]), int(kps[pid][1])), color=(0, 255, 0), thickness=2)

    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    return img

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def get_bbox(uv, hand_type):
    if (hand_type == "right"):
        s = slice(0, orig_root_idx["right"])
    elif (hand_type == "left"):
        s = slice(0, orig_root_idx["left"])
    elif (hand_type == "interacting"):
        s = slice(0, 41)

    x = max(min(uv[s, 0]) - 10, 0)
    y = max(min(uv[s, 1]) - 10, 0)

    x_max = min(max(uv[s, 0]) + 10, 255)
    y_max = min(max(uv[s, 1]) + 10, 255)

    # xmin, ymin, width, height
    return [
        float(max(0, x)), float(max(0, y)), float(x_max - x), float(y_max - y)
    ]




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

num_items = len(db.anns.keys())
NEW_SKELETON_IDS = [20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, \
41, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

print(f"Num {split}: {len(db.anns.keys())}")

output = {
    "images": [],
    "annotations": [],
    "categories": [{
      'supercategory': 'person',
      'id': 1,
      'name': 'person'
    }]
}

count = 0
for aid in db.anns.keys():
    ann = db.anns[aid]
    image_id = ann['image_id']
    img = db.loadImgs(image_id)[0]
    capture_idx = img['capture']
    seq_name = img['seq_name']
    cam_idx = img['camera']
    frame_idx = img['frame_idx']

    img_path = osp.join(img_root_path, split, img['file_name'])
    joint_param = joints[str(capture_idx)][str(frame_idx)]
    campos, camrot = np.array(cam_params[str(capture_idx)]['campos'][str(cam_idx)], dtype=np.float32), np.array(cam_params[str(capture_idx)]['camrot'][str(cam_idx)], dtype=np.float32)
    focal, princpt = np.array(cam_params[str(capture_idx)]['focal'][str(cam_idx)], dtype=np.float32), np.array(cam_params[str(capture_idx)]['princpt'][str(cam_idx)], dtype=np.float32)

    joint_world = np.array(joints[str(capture_idx)][str(frame_idx)]['world_coord'], dtype=np.float32)
    joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
    joint_2d = cam2pixel(joint_cam, focal, princpt)[:,:2]
        
    joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(joint_num*2)
    # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
    joint_valid[orig_joint_type['right']] *= joint_valid[orig_root_idx['right']]
    joint_valid[orig_joint_type['left']] *= joint_valid[orig_root_idx['left']]
    
    if (joint_valid[orig_root_idx['right']] == 0 and joint_valid[orig_root_idx["left"]] == 0):
        count += 1
        print("Both invalid..skipping")
        continue

    center = get_center(joint_2d, joint_valid)
    if (center[0] < 0 or center[1] < 0 or center[0] >= img['width'] or center[1] >= img['height']):
        count += 1
        print("Skipping because wrist off center")
        continue

    img = cv2.imread(img_path) 
    processed_img, x_offset, y_offset, scale = crop_and_center(img, joint_2d, joint_valid)

    joint_2d *= scale
    joint_2d[:, 0] -= x_offset
    joint_2d[:, 1] -= y_offset
    
    output_path = img_path.replace("InterHand2.6M_5fps_batch1", "InterHand2.6M_5fps_batch1_output")
    make_dirs(output_path)
    
    K = np.array([[focal[0], 0, princpt[0]], [0, focal[1], princpt[1]], [0, 0, 1.0]])
    joint_3d = reproject_to_3d(joint_2d, K, joint_cam[:, 2])

    joints_2d_proj = project_3D_points(K, joint_3d)
    # processed_img = vis_keypoints(processed_img, joints_2d_proj, skeleton, get_bbox(joint_2d, ann["hand_type"]), joint_valid)
    cv2.imwrite(output_path, processed_img)
    output["images"].append({
      "id": count,
      "width": 256,
      "height": 256,
      "file_name": output_path.replace("RawDatasets", "ProcessedDatasets"),
      "camera_param": {
          "focal": [float(K[0][0]), float(K[1][1])],
          "princpt": [float(K[0][2]), float(K[1][2])]
      }
    })
    print(joint_2d[NEW_SKELETON_IDS])
    input("? ")
    output["annotations"].append({
      "id": count,
      "image_id": count,
      "category_id": 1,
      "is_crowd": 0,
      "joint_img": joint_2d[NEW_SKELETON_IDS].tolist(),
      "joint_valid": joint_valid[NEW_SKELETON_IDS].tolist(),
      "hand_type": ann['hand_type'],
      "joint_cam": (joint_3d[NEW_SKELETON_IDS]).tolist(),
      "bbox": get_bbox(joint_2d, ann["hand_type"])
    })
    count += 1
    if (count % 100 == 0):
        percent = 100 * (float(count)/num_items)
        print("Idx: " + str(count) + ", percent: " + str(round(percent, 2)) + "%")
f = open("interhand_"+split+".json", "w")
json.dump(output, f)
f.close()
