import os
import pickle
import cv2
import numpy as np
import mxnet as mx
from tqdm import tqdm


# change this for other dataset
path = "data/lfw.bin"
image_size = (112,112)
outpath = "data/quality_data"


try:
    with open(path, 'rb') as f:
        bins, issame_list = pickle.load(f)  # py2
except UnicodeDecodeError as e:
    with open(path, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')  # py3

dataset_name = path.split("/")[-1].split(".")[0]
rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
outpath = os.path.join(outpath, dataset_name)
if not os.path.exists(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, "images"))

print("extract:", dataset_name)
pair_list = np.zeros((len(issame_list), 3), np.int16)

txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")

for idx in tqdm(range(len(bins))):
    _bin = bins[idx]
    img = mx.image.imdecode(_bin)
    if img.shape[1] != image_size[0]:
        img = mx.image.resize_short(img, image_size[0])

    cv2.imwrite(os.path.join(outpath, "images", "%05d.jpg" % idx), img.asnumpy())
    txt_file.write(os.path.join(rel_img_path, "%05d.jpg" % idx)+"\n")

    if idx % 2 == 0:
        pair_list[idx//2] = [idx, idx+1, issame_list[idx//2]]

txt_file.close()
np.savetxt(os.path.join(outpath, "pair_list.txt"), pair_list, fmt='%05d')
print("pair_list saved")
