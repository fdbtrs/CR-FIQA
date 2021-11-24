#!/usr/bin/env python
import argparse
import os
import glob
import math
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn import metrics

from collections import defaultdict
import gc

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from ERC.roc import get_eer_threshold

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--embeddings_dir', type=str,
                    default="/usr/quality_embeddings",
                    help='The dir save embeddings for each method and dataset, the diretory inside should be: {dataset}_{model}, e.g., IJBC_ArcFaceModel')
parser.add_argument('--quality_score_dir', type=str,
                    default="/usr/quality_score",
                    help='The dir save file of quality scores for each dataset and method, the file inside should be: {method}_{dataset}.txt, e.g., CRFIQAS_IJBC.txt')

parser.add_argument('--method_name', type=str,
                    default="Serfiq,BRISQUE,SDD,rankIQ,magface,FaceQnet,DeepIQA,PFE,rankIQA,CRFIQAL,CRFIQAS",
                    help='The evaluated image quality estimation method')
parser.add_argument('--models', type=str,
                    default="ArcFaceModel, ElasticFaceModel, MagFaceModel, CurricularFaceModel",
                    help='The evaluated FR model')
parser.add_argument('--eval_db', type=str,
                    default="adience,XQLFW,lfw,agedb_30,calfw,cplfw,cfp_fp",
                    help='The evaluated dataset')

parser.add_argument('--distance_metric', type=str,
                    default='cosine',
                    help='Cosine distance or euclidian distance')
parser.add_argument('--output_dir', type=str,
                    default="erc_plot_auc_test_all",
                    help='')

IMAGE_EXTENSION = '.jpg'

def load_quality(scores):
    quality={}
    with open(scores[0], 'r') as f:
        lines=f.readlines()
        for l in lines:
            scores = l.split()[1].strip()
            n = l.split()[0].strip()
            quality[n] = scores
    return quality

def load_quality_pair(pair_path, scores, dataset, args):
    pairs_quality = []
    quality=load_quality(scores)
    with open(pair_path, 'r') as f:
        lines=f.readlines()
        for idex in range(len(lines)):
            a= lines[idex].rstrip().split()[0]
            b= lines[idex].rstrip().split()[1]
            qlt=min(float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{a}{IMAGE_EXTENSION}"))),
                    float(quality.get(os.path.join(args.quality_score_dir, dataset, 'images', f"{b}{IMAGE_EXTENSION}"))))
            pairs_quality.append(qlt)
    return pairs_quality

def load_feat_pair(pair_path, root):
    pairs = {}
    with open(pair_path, 'r') as f:
        lines=f.readlines()
        for idex in range(len(lines)):
            a= lines[idex].rstrip().split()[0]
            b= lines[idex].rstrip().split()[1]
            is_same=int(lines[idex].rstrip().split()[2])
            feat_a=np.load(os.path.join(root, f"{a}.npy"))
            feat_b=np.load(os.path.join(root, f"{b}.npy"))
            pairs[idex] = [feat_a, feat_b, is_same]
    print("All features are loaded")
    return pairs

def distance_(embeddings0, embeddings1, dist="cosine"):
    # Distance based on cosine similarity
    if (dist=="cosine"):
        dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
        norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
        # shaving
        similarity = np.clip(dot / norm, -1., 1.)
        dist = np.arccos(similarity) / math.pi
    else:
        embeddings0 = sklearn.preprocessing.normalize(embeddings0)
        embeddings1 = sklearn.preprocessing.normalize(embeddings1)
        diff = np.subtract(embeddings0, embeddings1)
        dist = np.sum(np.square(diff), 1)

    return dist

def calc_score(embeddings0, embeddings1, actual_issame, subtract_mean=False, dist_type='cosine'):
    assert (embeddings0.shape[0] == embeddings1.shape[0])
    assert (embeddings0.shape[1] == embeddings1.shape[1])

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings0, embeddings1]), axis=0)
    else:
        mean = 0.

    dist = distance_(embeddings0, embeddings1, dist=dist_type)
    # sort in a desending order
    pos_scores =np.sort(dist[actual_issame == 1])
    neg_scores = np.sort(dist[actual_issame == 0])
    return pos_scores, neg_scores

def save_pdf(fnmrs_lists, method_labels, model, output_dir, fmr, db):
    fontsize = 20
    colors = ['green', 'black', 'orange', 'plum', 'cyan', 'gold', 'gray', 'salmon', 'deepskyblue', 'red', 'blue',
               'darkseagreen', 'seashell', 'hotpink', 'indigo', 'lightseagreen', 'khaki', 'brown', 'teal', 'darkcyan']
    STYLES = ['--', '-.', ':', 'v--', '^--', ',--', '<--', '>--', '1--',
             '-' ,'-' , '2--', '3--', '4--', '.--', 'p--', '*--', 'h--', 'H--', '+--', 'x--', 'd--', '|--', '---']
    unconsidered_rates = 100 * np.arange(0, 0.98, 0.05)

    fig, ax1 = plt.subplots()  # added
    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir)

    for i in range(len(fnmrs_lists)):
        print(fnmrs_lists[i])
        plt.plot(unconsidered_rates[:len(fnmrs_lists[i])], fnmrs_lists[i], STYLES[i], color=colors[i],
                 label=method_labels[i])
        auc_value =  metrics.auc( np.array(unconsidered_rates[:len(fnmrs_lists[i])]/100), np.array(fnmrs_lists[i]))
        with open(os.path.join(output_dir, db, str(fmr)+"_auc.txt"), "a") as f:
            f.write(db + ':' + model + ':' + method_labels[i] + ':' + str(auc_value) + '\n')
    plt.xlabel('Ratio of unconsidered image [%]')

    plt.xlabel('Ratio of unconsidered image [%]', fontsize=fontsize)
    plt.xlim([0, 98])
    plt.xticks(np.arange(0, 98, 10), fontsize=fontsize)
    plt.title(f"Testing on {db}, FMR={fmr}" + f" ({model})", fontsize=fontsize) # update : -3
    plt.ylabel('FNMR', fontsize=fontsize)

    axbox = ax1.get_position()
    fig.legend(bbox_to_anchor=(axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.22), prop=FontProperties(size=12),
               loc='lower center', ncol=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, db, db + '_' +str(fmr) +'_'+model + '.png'), bbox_inches='tight')

def getFNMRFixedTH(feat_pairs, qlts,  dist_type='cosine', desc=True):
    embeddings0, embeddings1, targets = [], [], []
    pair_qlt_list = []  # store the min qlt
    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])
        # convert into np
        np_feat_a = np.asarray(feat_a, dtype=np.float64)
        np_feat_b = np.asarray(feat_b, dtype=np.float64)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)
        targets.append(ab_is_same)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(-1, )
    qlts = np.array(qlts)
    if (desc):
        qlts_sorted_idx = np.argsort(qlts)
    else:
        qlts_sorted_idx = np.argsort(qlts)[::-1]

    num_pairs = len(targets)
    unconsidered_rates = np.arange(0, 0.98, 0.05)

    fnmrs_list_2 = []
    fnmrs_list_3 = []
    fnmrs_list_4 = []
    for u_rate in unconsidered_rates:
        hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs):]
        pos_dists, neg_dists = calc_score(embeddings0[hq_pairs_idx], embeddings1[hq_pairs_idx], targets[hq_pairs_idx], dist_type=dist_type)
        fmr100_th, fmr1000_th, fmr10000_th = get_eer_threshold(pos_dists, neg_dists, ds_scores=True)

        g_true = [g for g in pos_dists if g < fmr100_th]
        fnmrs_list_2.append(1- len(g_true)/(len(pos_dists)))
        g_true = [g for g in pos_dists if g < fmr1000_th]
        fnmrs_list_3.append(1 - len(g_true) / (len(pos_dists)))
        g_true = [g for g in pos_dists if g < fmr10000_th]
        fnmrs_list_4.append(1 - len(g_true) / (len(pos_dists)))

    return fnmrs_list_2,fnmrs_list_3,fnmrs_list_4,unconsidered_rates


def perform_1v1_quality_eval(args):
 d = args.eval_db.split(',')

 for dataset in d:
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-2)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir, dataset, str(1e-2)+"_auc.txt"))
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-3)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir, dataset, str(1e-3)+"_auc.txt"))
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-4)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir,  dataset, str(1e-4)+"_auc.txt"))

 models=args.models.split(',')
 for model in models:
  for dataset in d:
    method_names = args.method_name.split(',')
    fnmrs_list_2=[]
    fnmrs_list_3=[]
    fnmrs_list_4=[]
    method_labels=[]

    if (not os.path.isdir(os.path.join(args.output_dir, dataset, 'fnmr'))):
        os.makedirs(os.path.join(args.output_dir, dataset, 'fnmr'))

    unconsidered_rates = np.arange(0, 0.98, 0.05)

    for method_name in method_names:
        print(f"----process {model} {dataset} {method_name}-----------")
        desc = False if method_name == 'PFE' else True

        feat_pairs = load_feat_pair(os.path.join(args.quality_score_dir, dataset, 'pair_list.txt'),
                                   os.path.join(args.embeddings_dir, f"{dataset}_{model}"))

        quality_scores = load_quality_pair(os.path.join(args.quality_score_dir, dataset, 'pair_list.txt')
                                          [os.path.join(args.quality_score_dir, dataset, f"{method_name}_{dataset}.txt")],
                                          dataset, args)

        fnmr2, fnmr3, fnmr4, unconsidered_rates = getFNMRFixedTH(feat_pairs, quality_scores, dist_type=args.distance_metric, desc=desc)
        fnmrs_list_2.append(fnmr2)
        fnmrs_list_3.append(fnmr3)
        fnmrs_list_4.append(fnmr4)
        method_labels.append(f"{method_name}")

        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr2.npy"), fnmr2)
        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr3.npy"), fnmr3)
        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr4.npy"), fnmr4)

    save_pdf(fnmrs_list_2, method_labels, model=model, output_dir=args.output_dir, fmr =1e-2, db=dataset)
    save_pdf(fnmrs_list_3, method_labels, model=model, output_dir=args.output_dir, fmr =1e-3, db=dataset)
    save_pdf(fnmrs_list_4, method_labels, model=model, output_dir=args.output_dir, fmr =1e-4, db=dataset)

def main():
    args = parser.parse_args()
    perform_1v1_quality_eval(args)

if __name__ == '__main__':
    main()
