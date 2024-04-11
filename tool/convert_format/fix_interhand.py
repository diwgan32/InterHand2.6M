## Assuming "data" has already been read and holds data
import numpy as np

NEW_SKELETON_IDS = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, \
41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]

for i in range(len(data["images"])):
    data["annotations"][i]["joint_img"] = (np.array(data["annotations"][i]["joint_img"])[NEW_SKELETON_IDS]).tolist()
    data["annotations"][i]["joint_valid"] = (np.array(data["annotations"][i]["joint_valid"])[NEW_SKELETON_IDS]).tolist()
    data["annotations"][i]["joint_cam"] = (np.array(data["annotations"][i]["joint_cam"])[NEW_SKELETON_IDS]).tolist()

f = open("/home/ubuntu/RawDatasets/interhand_training_fixed.json", "w")
json.dump(data, f)
f.close()

