import os
import cv2
import numpy as np
import shutil
from skimage import transform as trans
from tqdm import tqdm


# change this for other dataset
path = "/data/fboutros/IJB_release/IJB_release/IJBB"
image_size = (112,112)
outpath = "/data/maklemt/quality_data"

ref_lmk = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
ref_lmk[:, 0] += 8.0

dataset_name = path.split("/")[-1]
rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
outpath = os.path.join(outpath, dataset_name)
if not os.path.exists(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, "images"))

print("extract:", dataset_name)

img_path = os.path.join(path, "loose_crop")
img_list_path = os.path.join(path, "meta", f"{dataset_name.lower()}_name_5pts_score.txt")
img_list = open(img_list_path)
files_list = img_list.readlines()

txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")

for img_index, each_line in tqdm(enumerate(files_list), total=len(files_list)):
    name_lmk_score = each_line.strip().split(' ')
    img_name = os.path.join(img_path, name_lmk_score[0])
    img = cv2.imread(img_name)
    lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                    dtype=np.float32)
    lmk = lmk.reshape((5, 2))

    assert lmk.shape[0] == 5 and lmk.shape[1] == 2

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, ref_lmk)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(outpath, "images", name_lmk_score[0]), img)
    txt_file.write(os.path.join(rel_img_path, name_lmk_score[0])+"\n")


txt_file.close()
shutil.copy(
    os.path.join(path, "meta", f"{dataset_name.lower()}_template_pair_label.txt"),
    os.path.join(outpath, "pair_list.txt")
)
print("pair_list saved")
