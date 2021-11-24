import os
import cv2

path = "./data/XQLFW"
outpath = "./data/quality_data"

dataset_name = path.split("/")[-1]
rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
outpath = os.path.join(outpath, dataset_name)
if not os.path.exists(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, "images"))

align_path = os.path.join(path, "xqlfw_aligned_112")


def copy_img(person, img_id):
    img_name = f"{person}_{str(img_id).zfill(4)}.jpg"
    tmp_path = os.path.join(align_path, person, img_name)
    img = cv2.imread(tmp_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(outpath, "images", img_name), img)
    return img_name


def create_xqlfw_pairs(pairs_filename):
    """ reads xqlfw pairs.txt and creates pair_list.txt 
    and image_path_list.txt of required format
    :param pairs_filename: path to pairs.txt
    """
    txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")
    pair_list = open(os.path.join(outpath, "pair_list.txt"), "w")

    f = open(pairs_filename, 'r')
    for line in f.readlines()[1:]:
        pair = line.strip().split()
        if len(pair) == 3:
            img_name1 = copy_img(pair[0], pair[1])
            img_name2 = copy_img(pair[0], pair[2])
        else:
            img_name1 = copy_img(pair[0], pair[1])
            img_name2 = copy_img(pair[2], pair[3])

        txt_file.write(os.path.join(rel_img_path, img_name1)+"\n")
        txt_file.write(os.path.join(rel_img_path, img_name2)+"\n")
        pair_list.write(f"{img_name1} {img_name2} {int(len(pair)==3)}\n")
            

    f.close()
    txt_file.close()
    pair_list.close()

create_xqlfw_pairs("/data/maklemt/XQLFW/xqlfw_pairs.txt")
print("XQLFW successfully extracted")