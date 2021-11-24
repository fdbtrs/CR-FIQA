# coding: utf-8

import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import sys
import warnings

from tqdm import tqdm


sys.path.insert(0, "../")
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--image-path', default='/data/fboutros/IJB_release/IJB_release/IJBB', type=str, help='')
parser.add_argument('--template-path', default='/data/maklemt/quality_embeddings', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--embedding-size', default=512, type=int, help='')
parser.add_argument('--network', default='iresnet50', type=str, help='')
parser.add_argument('--quality-model', default='Serfiq', type=str, help='')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
image_path = args.image_path
template_path = os.path.join(args.template_path, f"{target}_{model_path}_raw")
result_dir = args.result_dir
gpu_id = 0
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
batch_size = args.batch_size
embedding_size = args.embedding_size
quality_model = args.quality_model

if model_path == "ArcFaceModel":
    import mxnet as mx
    batch_size = 32 # because of hardware limitations

import cv2
import numpy as np
import torch
from skimage import transform as trans


def get_backbone(modelname):
    if modelname == "ArcFaceModel":
        prefix = "/home/fboutros/PR/pretrained-model/model-r100-ii/model"
        epoch = "0000"
        if gpu_id>=0:
            ctx = mx.gpu(gpu_id)
        else:
            ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        backbone = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        backbone.bind(data_shapes=[('data', (1, 3, 112, 112))])
        backbone.set_params(arg_params, aux_params)
    elif modelname == "CurricularFaceModel":
        from backbones.model_irse import IR_SE_101, IR_101
        prefix = "/home/fboutros/FR-vulnerability/CurricularFace_Backbone"
        epoch = "295672"
        weight = torch.load(os.path.join(prefix,"CurricularFace_Backbone.pth"))
        backbone = IR_101([112,112]).to(f"cuda:{gpu_id}")
        backbone.load_state_dict(weight)
    elif modelname == "ElasticFaceModel":
        from backbones.iresnet import iresnet100
        prefix = "/home/fboutros/FR-vulnerability/ElasticFace"
        epoch = "295672"
        weight = torch.load(os.path.join(prefix,epoch+"backbone.pth"))
        backbone = iresnet100().to(f"cuda:{gpu_id}")
        backbone.load_state_dict(weight)
    elif modelname == "MagFaceModel":
        from backbones.mag_network_inf import builder_inf
        prefix = "/home/maklemt/image-quality"
        epoch = "00025"
        backbone = builder_inf(os.path.join(prefix,"magface_epoch_"+epoch+".pth"), "iresnet100", 512)
    else:
        print("Unknown model")
        exit()
    return backbone

class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        self.image_size = (112, 112)
        backbone = get_backbone(prefix)
        if prefix == "ArcFaceModel":
            self.model = backbone
        else:
            self.model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
            self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.modelname = prefix

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f"IJBTest.png", img)
        # exit()
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[0], self.image_size[1]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        if self.modelname == "ArcFaceModel":
            data = mx.nd.array(batch_data)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            emb = self.model.get_outputs()[0].asnumpy()
            emb = emb.reshape([self.batch_size, 2 * emb.shape[1]])
            return emb

        imgs = torch.Tensor(batch_data).cuda()
        if self.modelname == "MagFaceModel":
            imgs.div_(255)
        else:
            imgs.div_(255).sub_(0.5).div_(0.5)
        
        if self.modelname == "CurricularFaceModel":
            feat, _ = self.model(imgs)
        else:
            feat = self.model(imgs)
        feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias



def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


def read_quality_scores(path):
    quality_scores = pd.read_csv(path, sep=' ', header=None).values
    score_dict = { qs[0].split("/")[-1] : float(qs[1]) for qs in quality_scores }
    print("Quality scores loaded")
    return score_dict


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def get_image_feature(img_path, files_list, model_path, epoch, gpu_id, quality_dict):
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    quality_scores=[]
    batch = 0
    img_feats = np.empty((len(files), embedding_size*2), dtype=np.float32)

    """
    for img_index, each_line in tqdm(enumerate(files), total=len(files)):
        name_lmk_score = each_line.strip().split(' ')
        template_name = os.path.join(template_path, name_lmk_score[0].replace('.jpg', '.npy'))
        img_feats[img_index] = np.load(template_name)
        qual_key = name_lmk_score[0]
        quality_scores.append(quality_dict[qual_key])
        faceness_scores.append(name_lmk_score[-1])
    """

    # extract embeddings with model
    batch_data = np.empty((2 * batch_size, 3, img_size[0], img_size[1]))
    embedding = Embedding(model_path, data_shape, batch_size)
    for img_index, each_line in enumerate(files[:len(files) - rare_size]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)

        batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
        batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
        if (img_index + 1) % batch_size == 0:
            print('batch', batch)
            img_feats[batch * batch_size:batch * batch_size +
                                         batch_size][:] = embedding.forward_db(batch_data)
            batch += 1
        qual_key = img_name.split("/")[-1]
        quality_scores.append(quality_dict[qual_key])
        faceness_scores.append(name_lmk_score[-1])

    batch_data = np.empty((2 * rare_size, 3, img_size[0], img_size[1]))
    embedding = Embedding(model_path, data_shape, rare_size)
    for img_index, each_line in enumerate(files[len(files) - rare_size:]):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        input_blob = embedding.get(img, lmk)
        batch_data[2 * img_index][:] = input_blob[0]
        batch_data[2 * img_index + 1][:] = input_blob[1]
        if (img_index + 1) % rare_size == 0:
            print('batch', batch)
            img_feats[len(files) -
                      rare_size:][:] = embedding.forward_db(batch_data)
            batch += 1
        qual_key = img_name.split("/")[-1]
        quality_scores.append(quality_dict[qual_key])
        faceness_scores.append(name_lmk_score[-1])
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    quality_scores = np.array(quality_scores).astype(np.float32)
    # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores, quality_scores


def image2template_feature(img_feats=None, templates=None, medias=None ,quality_scores=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        
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
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data
assert target == 'IJBC' or target == 'IJBB'
img_size = (112, 112)

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % image_path,
                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))


# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))


# # Step 2: Get Image Features
# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = '%s/loose_crop' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
# files_list = divideIntoNstrand(files, rank_size)
files_list = files

quality_path = os.path.join("quality_scores", f"{quality_model}_{target}.txt")
quality_dict = read_quality_scores(quality_path)

# img_feats
# for i in range(rank_size):
img_feats, faceness_scores, quality_scores= get_image_feature(img_path, files_list,
                                               model_path, 0, gpu_id, quality_dict)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# # Step3: Get Template Features
# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    # img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                     2] + img_feats[:, img_feats.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    qs = quality_scores
    qs = (qs - min(qs)) / (max(qs) - min(qs))
    if quality_model == "PFE":
        qs = 1 - qs
    print(img_input_feats.shape, qs.shape)
    img_input_feats = (img_input_feats * qs[:, np.newaxis]) / sum(qs)
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias, quality_scores)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores
# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

save_path = os.path.join("results", f"{model_path}_{quality_model}")
if not os.path.exists(save_path):
    os.makedirs(save_path)
file_path = os.path.join(save_path, target + "_results.txt")


# save_path = os.path.join(result_dir, args.job)
# save_path = result_dir + '/%s_result' % target

model_name = f"{model_path}_{quality_model}"
score_save_path = os.path.join("tmp", model_name)
if not os.path.exists(score_save_path):
    os.makedirs(score_save_path)

score_save_file = os.path.join(score_save_path, "%s.npy" % target.lower())
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table
files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target))
print(tpr_fpr_table)


## save prettytable
data = tpr_fpr_table.get_string()
f = open(file_path, "w")
f.write(data)
f.close()
