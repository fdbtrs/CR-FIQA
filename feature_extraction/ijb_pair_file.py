import os
import pandas as pd
import numpy as np
import sklearn.preprocessing
from tqdm import tqdm
from multiprocessing import Process, Queue

dataset_name = "IJBC"
model = "ArcFaceModel"          # ArcFaceModel, CurricularFaceModel, ElasticFaceModel, MagFaceModel
quality_model = "rankIQA"        # Serfiq, BRISQUE, SDD, rankIQ, PFE, magface, FaceQnet, DeepIQA, CRFIQAL
threads = 20
quality_score_path = os.path.join("quality_scores", f"{quality_model}_{dataset_name}.txt")
path = os.path.join("/data/fboutros/IJB_release/IJB_release", dataset_name)
raw_feature_path = os.path.join("/data/maklemt/quality_embeddings", dataset_name + "_" + model + "_raw")
outpath = raw_feature_path[:-4]
if not os.path.exists(outpath):
    os.makedirs(outpath)

tid_mid_path = os.path.join(path, "meta", f"{dataset_name.lower()}_face_tid_mid.txt")
pair_path = os.path.join(path, "meta", f"{dataset_name.lower()}_template_pair_label.txt")

def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    print("Template pair list loaded")
    return t1, t2, label


def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    img_names = ijb_meta[:, 0]
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    print("Tid mid list loaded")
    return img_names, templates, medias


def load_raw_templates(files):
    features = np.zeros((len(files), 1024))
    for i, file in enumerate(files):
        filepath = os.path.join(raw_feature_path, file.replace(".jpg", ".npy"))
        features[i] = np.load(filepath)
    return features


def aggregate_templates():
    imgs_names, templates, medias = read_template_media_list(tid_mid_path)

    unique_templates = np.unique(templates)

    for uqt in tqdm(unique_templates, total=len(unique_templates)):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = load_raw_templates(imgs_names[ind_t])
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats, dtype=np.float32)
        aggregated_feature = sklearn.preprocessing.normalize(np.sum(media_norm_feats, axis=0))
        np.save(os.path.join(outpath, f"{uqt}.npy"), aggregated_feature[0])


aggregate_templates()



def load_quality_scores(path):
    quality_scores = pd.read_csv(path, sep=' ', header=None).values
    score_dict = { qs[0].split("/")[-1] : float(qs[1]) for qs in quality_scores }
    print("Quality scores loaded")
    return score_dict

imgs_names, templates, _ = read_template_media_list(tid_mid_path)
score_dict = load_quality_scores(quality_score_path)

def get_min_score(person):
    (ind_t1,) = np.where(templates == person)
    scores = []
    for img_name in imgs_names[ind_t1]:
        scores.append(score_dict[img_name])
    return max(scores) if quality_model == "PFE" else min(scores)


def get_score_part(t1s, t2s, labels, queue):
    q_pair_list = ""
    for t1, t2, label in tqdm(zip(t1s, t2s, labels), total=len(t1s)):
        min_score_t1 = get_min_score(t1)
        min_score_t2 = get_min_score(t2)
        if quality_model == "PFE":
            min_score = max([min_score_t1, min_score_t2])
        else:
            min_score = min([min_score_t1, min_score_t2])

        q_pair_list += f"{t1} {t2} {label} {min_score}\n"

    queue.put(q_pair_list)


def create_quality_list():
    t1s, t2s, labels = read_template_pair_list(pair_path)
    part_idx = len(t1s) // threads
    print(f"Quality estimation model: {quality_model}\n{threads+1} Threads")

    q = Queue()
    processes = []

    for idx in range(threads):
        t1_part = t1s[idx*part_idx:(idx+1)*part_idx]
        t2_part = t2s[idx*part_idx:(idx+1)*part_idx]
        label_part = labels[idx*part_idx:(idx+1)*part_idx]
        p = Process(target=get_score_part, args=(t1_part, t2_part, label_part, q))
        processes.append(p)
        p.start()

    t1_part = t1s[threads*part_idx:]
    t2_part = t2s[threads*part_idx:]
    label_part = labels[threads*part_idx:]
    p = Process(target=get_score_part, args=(t1_part, t2_part, label_part, q))
    processes.append(p)
    p.start()
    

    save_path = os.path.join("/data/maklemt/quality_data", dataset_name)
    pair_list = open(os.path.join(save_path, f"quality_pair_list_{quality_model}_{dataset_name}.txt"), "w")

    for p in processes:
        ret = q.get() # will block
        pair_list.write(ret)
    for p in processes:
        p.join()

    pair_list.close()


create_quality_list()