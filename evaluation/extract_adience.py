import os
import cv2
import numpy as np
from tqdm import tqdm
from align_trans import norm_crop

from mtcnn import MTCNN
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)#, device_count = {'GPU': 0}
)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list= '0'
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


imgs_path = "/data/maklemt/adience"
outpath = "/data/maklemt/quality_data"


def find_central_face(img, keypoints):
    # if multiple faces are detected, select the face most in the middle of the image
    mid_face_idx = 0
    if len(keypoints) > 1:
        img_mid_point = np.array([img.shape[1]//2, img.shape[0]//2])    # [x (width), y (height)]
        noses = np.array([keypoint['keypoints']['nose'] for keypoint in keypoints])
        distances = np.linalg.norm(noses - img_mid_point, axis=1)       # calculate distance between nose and img mid point
        mid_face_idx = np.argmin(distances)

    facial5points = [keypoints[mid_face_idx]['keypoints']['left_eye'], keypoints[mid_face_idx]['keypoints']['right_eye'], 
                    keypoints[mid_face_idx]['keypoints']['nose'], keypoints[mid_face_idx]['keypoints']['mouth_left'], 
                    keypoints[mid_face_idx]['keypoints']['mouth_right']]
    return np.array(facial5points)



detector = MTCNN(min_face_size=20, steps_threshold=[0.6, 0.7, 0.9], scale_factor=0.85)
skipped_imgs = []

dataset_name = imgs_path.split("/")[-1]
rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
outpath = os.path.join(outpath, dataset_name)
if not os.path.exists(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, "images"))

print("extract:", dataset_name)

txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")
img_files = os.listdir(imgs_path)

for img_index, img_file in tqdm(enumerate(img_files), total=len(img_files)):
    if img_file.split('.')[-1] != "jpg":
        continue
    img_path = os.path.join(imgs_path, img_file)
    img = cv2.imread(img_path)

    keypoints = detector.detect_faces(img)
    if len(keypoints) == 0:
        skipped_imgs.append(img_file)
        continue
    facial5points = find_central_face(img, keypoints)
    warped_face = norm_crop(img, landmark=facial5points, createEvalDB=True)
    
    img = cv2.cvtColor(warped_face, cv2.COLOR_RGB2BGR)
    person = img_file.split('.')[1]
    new_img_name = f"p_{person}_img_{img_index}.jpg"

    cv2.imwrite(os.path.join(outpath, "images", new_img_name), img)
    txt_file.write(os.path.join(rel_img_path, new_img_name)+"\n")

txt_file.close()


print("creating pair list...")
pair_list = open(os.path.join(outpath, "pair_list.txt"), "w")
aligned_img_path = os.listdir(os.path.join(outpath, "images"))

for img_index1, img_file1 in enumerate(aligned_img_path):
    if img_file1.split('.')[-1] != "jpg":
        continue
    person1 = img_file1.split('_')[1]

    for img_index2, img_file2 in enumerate(aligned_img_path[img_index1+1:]):
        if img_file2.split('.')[-1] != "jpg":
            continue
        person2 = img_file2.split('_')[1]
        genuine = person1 == person2
        pair_list.write(f"{img_file1.split('.')[0]} {img_file2.split('.')[0]} {int(genuine)}\n")

pair_list.close()
print("pair_list saved")
print("No faces detected in:")
print(skipped_imgs)
print("Total amount of images with no detected face: ", len(skipped_imgs))
