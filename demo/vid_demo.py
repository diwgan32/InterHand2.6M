# Runs the network on a video
# and renders hand pose on new video file
# Expects 2d bbox's from wrnch

import sys
import os
import json
import os.path as osp
import argparse
import numpy as np
import cv2
import video_utils
import sys
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, load_skeleton, process_bbox, generate_patch_image, transform_input_to_output_space, trans_point2d
from utils.vis import vis_keypoints, vis_3d_keypoints
RIGHT_WRNCH_WRIST_ID = 10
LEFT_WRNCH_WRIST_ID = 15
NUM_WRNCH_HAND_JOINTS = 21
NUM_INTERHAND_JOINTS = 21
DISP = False
JOINT_TYPE = {'right': np.arange(0,NUM_INTERHAND_JOINTS), 'left': np.arange(NUM_INTERHAND_JOINTS,NUM_INTERHAND_JOINTS*2)}
SIDE = "right"
# HanCo validated training dataset, with greenscreens swapped out
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--wrnch_path', type=str, dest='wrnch_path')
    parser.add_argument('--video_path', type=str, dest='video_path')
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def read_wrnch(file_path):
    f = open(file_path,)
    data = json.load(f)
    return data

def get_hand_pose(wrnch_data, frame_no, side="left"):
    frame = wrnch_data["frames"][frame_no]
    if ("persons" not in frame or len(frame["persons"]) == 0):
        print("No person")
        return None
    person = frame["persons"][0]
    if ("hand_pose" not in person):
        print("No hand pose")
        return None
    hand_pose = person["hand_pose"]
    if (side not in hand_pose):
        print("No hand")
        return None
    return hand_pose

def get_hand_bbox(wrnch_data, frame_no, side="left"):
    hand_pose = get_hand_pose(wrnch_data, frame_no, side)
    if (hand_pose is None):
        return None
    return hand_pose[side]["bbox"]

def get_small_hand_bbox(wrnch_data, frame_no, side="left"):
    hand_pose = get_hand_pose(wrnch_data, frame_no, side)
    if (hand_pose is None):
        return None
    joints = np.array(hand_pose[side]["joints"])
    x_vals = joints[::2]
    y_vals = joints[1::2]

    x_min = np.min(x_vals[x_vals > 0])
    y_min = np.min(y_vals[y_vals > 0])
    x_max = np.max(x_vals[x_vals > 0])
    y_max = np.max(y_vals[y_vals > 0])

    return {
        "minX": max(x_min - .03, 0),
        "minY": max(y_min - .03, 0),
        "height": min(y_max - y_min + .06, 1.0),
        "width": min(x_max - x_min  + .06, 1.0)
    }

def get_wrist_pos(wrnch_data, frame_no, side="L"):
    frame = wrnch_data["frames"][frame_no]
    if ("persons" not in frame or len(frame["persons"]) == 0):
        return None
    person = frame["persons"][0]
    if ("pose2d" not in person):
        return None
    pose2d = person["pose2d"]
    if ("joints" not in pose2d):
        return None
    WRNCH_WRIST_ID = LEFT_WRNCH_WRIST_ID if side == "L" else RIGHT_WRNCH_WRIST_ID
    wrist = np.array([
        pose2d["joints"][WRNCH_WRIST_ID * 2],
        pose2d["joints"][WRNCH_WRIST_ID * 2 + 1]
    ])
    if (wrist[0] < 0 or wrist[1] < 0):
        return None
    return wrist

def crop_and_center(imgInOrg, center, box_size):
    shape = imgInOrg.shape

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

    new_img = np.zeros((box_size, box_size, 3), dtype=np.uint8)
    
    new_img[y_min_n:y_min_n+h, x_min_n:x_min_n+w] = \
        imgInOrg[y_min_o:y_max_o, x_min_o:x_max_o]
    new_img = cv2.resize(new_img, (256, 256))
    scale = float(256)/box_size
    return new_img, x_min_o, y_min_o, scale

def get_cropped_image(img, wrnch_data, frame_no, side="left"):
    hand_bbox = get_hand_bbox(wrnch_data, frame_no, side)
    if (hand_bbox is None):
        return None, None, None, None

    wrist = get_wrist_pos(wrnch_data, frame_no, "L" if side == "left" else "R")
    if (wrist is None):
        return None, None, None, None
    wrist[0] *= img.shape[1]
    wrist[1] *= img.shape[0]
    # Pick largest dim to place square box, and provide some buffer
    box_size = max(hand_bbox["height"], hand_bbox["width"])
    box_size = None, None, None, None
    if (hand_bbox["height"] > hand_bbox["width"]):
        box_size = hand_bbox["height"] * img.shape[0] + 10
    else:
        box_size = hand_bbox["width"] * img.shape[1] + 10
    cropped_img, x_offset, y_offset, scale = crop_and_center(img, wrist, int(box_size))
    return cropped_img, x_offset, y_offset, scale

def load_model(args):
    # snapshot load
    model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model('test', 21)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    # argument parsing
    args = parse_args()
    cap = cv2.VideoCapture(args.video_path)
    wrnch_data = read_wrnch(args.wrnch_path)
    frame_no = 0
    is_portrait = video_utils.isVideoPortrait(args.video_path)
    skeleton_loc = '/home/ubuntu/Combined/skeleton.txt'
    skeleton = load_skeleton(osp.join(skeleton_loc), NUM_INTERHAND_JOINTS*2)
    writer = cv2.VideoWriter(
        filename=args.output_path,
        fps=30,
        fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        frameSize=(256, 256)
    )
    transform = transforms.ToTensor()
    model = load_model(args)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #if is_portrait:
        #    frame = cv2.transpose(frame, frame)
        #    frame = cv2.flip(frame, 1)
        cropped_img, x_offset, y_offset, scale = get_cropped_image(frame, wrnch_data, frame_no, SIDE)
        if (frame_no % 10 == 0):
            print(f"Frame no: {frame_no}")
        if (cropped_img is None):
            frame_no += 1
            continue

        hand_bbox = get_small_hand_bbox(wrnch_data, frame_no, SIDE)
        if (hand_bbox is None):
            frame_no += 1
            continue
        hand_bbox["height"] *= frame.shape[0]
        hand_bbox["width"] *= frame.shape[1]
        hand_bbox["minX"] *= frame.shape[1]
        hand_bbox["minY"] *= frame.shape[0]

        hand_bbox["minX"] -= x_offset
        hand_bbox["minY"] -= y_offset

        hand_bbox["height"] *= scale
        hand_bbox["width"] *= scale
        hand_bbox["minX"] *= scale
        hand_bbox["minY"] *= scale

        bbox = [
            int(hand_bbox["minX"]),
            int(hand_bbox["minY"]),
            hand_bbox["width"],
            hand_bbox["height"]
        ]

        if (DISP):
            cropped_img = cv2.rectangle(
                cropped_img,
                (int(hand_bbox["minX"]), int(hand_bbox["minY"])),
                (int(hand_bbox["minX"] + hand_bbox["width"]), int(hand_bbox["minY"] + hand_bbox["height"])),
                (0, 0, 0),
                2
            )

            cv2.imshow("Test", cropped_img)
            cv2.waitKey(0)

        original_img = cropped_img
        original_img_height, original_img_width = original_img.shape[:2]

        bbox = process_bbox(bbox, (original_img_height, original_img_width, original_img_height))
        
        img, trans, inv_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]

        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')
        img = img[0].cpu().numpy().transpose(1,2,0) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        joint_coord = out['joint_coord'][0].cpu().numpy() # x,y pixel, z root-relative discretized depth
        rel_root_depth = out['rel_root_depth'][0].cpu().numpy() # discretized depth
        hand_type = out['hand_type'][0].cpu().numpy() # handedness probability

        # restore joint coord to original image space and continuous depth space
        joint_coord[:,0] = joint_coord[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        joint_coord[:,1] = joint_coord[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        joint_coord[:,:2] = np.dot(inv_trans, np.concatenate((joint_coord[:,:2], np.ones_like(joint_coord[:,:1])),1).transpose(1,0)).transpose(1,0)
        joint_coord[:,2] = (joint_coord[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

        # restore right hand-relative left hand depth to continuous depth space
        rel_root_depth = (rel_root_depth/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

        # right hand root depth == 0, left hand root depth == rel_root_depth
        joint_coord[JOINT_TYPE[SIDE],2] += rel_root_depth

        # handedness
        joint_valid = np.zeros((NUM_INTERHAND_JOINTS*2), dtype=np.float32)
        right_exist = False
        if hand_type[0] > 0.5: 
            right_exist = True
            joint_valid[JOINT_TYPE['right']] = 1
        left_exist = False
        if hand_type[1] > 0.5:
            left_exist = True
            joint_valid[JOINT_TYPE['left']] = 1

        filename = f"output_{frame_no}.jpg"
        # visualize joint coord in 2D space
        vis_img = original_img.copy()[:,:,::-1].transpose(2,0,1)
        vis_img = vis_keypoints(vis_img, joint_coord, joint_valid, skeleton, filename, save_path=None, bbox=bbox)
        cv2_img = np.array(vis_img.convert("RGB"))
        writer.write(cv2_img[:, :, ::-1])

        frame_no += 1
    cap.release()
    writer.release()
    

