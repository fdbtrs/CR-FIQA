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

from roc import get_eer_threshold

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--embeddings_dir', type=str,
                    default="/usr/quality_embeddings",
                    help='The dir save embeddings for each method and dataset, the diretory inside should be: {dataset}_{model}, e.g., IJBC_ArcFaceModel')
parser.add_argument('--quality_score_dir', type=str,
                    default="/usr/quality_score",
                    help='The dir save file of quality scores for each dataset and method, the file inside should be: {method}_{dataset}.txt, e.g., CRFIQAS_IJBC.txt')

parser.add_argument('--models', type=str,
                    default="ArcFaceModel, ElasticFaceModel, MagFaceModel, CurricularFaceModel",
                    help='The evaluated FR model')
parser.add_argument('--eval_db', type=str,
                    default="IJBC",
                    help='The evaluated dataset')

parser.add_argument('--output_dir', type=str,
                    default="erc_plot_auc_test_all",
                    help='')
parser.add_argument('--method_name', type=str,
                    default="Serfiq,BRISQUE,SDD,rankIQ,magface,FaceQnet,DeepIQA,PFE,rankIQA,CRFIQAL,CRFIQAS"
                    help='The evaluated image quality estimation method')

parser.add_argument('--distance_metric', type=str,
                    default='cosine',
                    help='Euclidian Distance or Cosine Distance.')
parser.add_argument('--feat_size', type=int,
                    default=1024,
                    help='The size of extracted features')

def load_all_features(root):
    all_features = defaultdict()
    for feature_path in tqdm(glob.glob(os.path.join(root, '*.npy'))):
        feat = np.load(feature_path)
        all_features[os.path.basename(feature_path)] = feat
    print("All features are loaded")
    return all_features

def load_ijbc_pairs_features(pair_path, all_features, hq_pairs_idx, feature_size=1024):
    with open(pair_path, 'r') as f:
        lines=f.readlines()

    # build two empty embeddings matrix
    embeddings_0, embeddings_1 = np.empty([hq_pairs_idx.shape[0], feature_size]), np.empty([hq_pairs_idx.shape[0], feature_size])
    # load embeddings based on the needed pairs
    for indx in tqdm(range(hq_pairs_idx.shape[0])):
        real_index = hq_pairs_idx[indx]
        split_line = lines[real_index].split()
        feat_a = all_features[(split_line[0] + '.npy')]
        feat_b = all_features[(split_line[1] + '.npy')]
        embeddings_0[indx] = np.asarray(feat_a, dtype=np.float64)
        embeddings_1[indx] = np.asarray(feat_b, dtype=np.float64)

    return embeddings_0, embeddings_1

def load_ijbc_pairs_quality(pair_path):
    with open(pair_path, 'r') as f:
        lines=f.readlines()
    pairs_quality, targets = [], []

    for idex, line in enumerate(tqdm(lines)):
        split_line = line.split()
        pairs_quality.append(float(split_line[3]))  # quality score
        targets.append(int(split_line[2]))   # imposter or genuine
    targets = np.vstack(targets).reshape(-1, )
    print('Loaded quality score and target for each pair')
    return targets, np.array(pairs_quality)

def save_pdf(fnmrs_lists, method_labels, model, output_dir, fmr, db):
    fontsize = 20
    colors = ['green', 'black', 'orange', 'plum', 'cyan', 'gold', 'gray', 'salmon', 'deepskyblue', 'red', 'blue',
               'darkseagreen', 'seashell', 'hotpink', 'indigo', 'lightseagreen', 'khaki', 'brown', 'teal', 'darkcyan']
    STYLES = ['--', '-.', ':', 'v--', '^--', ',--', '<--', '>--', '1--',
             '-' ,'-' , '2--', '3--', '4--', '.--', 'p--', '*--', 'h--', 'H--', '+--', 'x--', 'd--', '|--', '---']

    unconsidered_rates = 100 * np.arange(0, 0.98, 0.05)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    if (not os.path.isdir(os.path.join(output_dir, 'plots', db))):
        os.makedirs(os.path.join(output_dir, 'plots', db))

    for i in range(len(fnmrs_lists)):
        ax1.plot(unconsidered_rates[:len(fnmrs_lists[i])], fnmrs_lists[i], STYLES[i], color=colors[i],
                 label=method_labels[i], linewidth=4, markersize=12)
        auc =  metrics.auc( np.array(unconsidered_rates[:len(fnmrs_lists[i])]/100),np.array(fnmrs_lists[i]))
        with open(os.path.join(output_dir, db, str(fmr)+"_auc.txt"), "a") as f:
            f.write(db + ':' + model + ':' + method_labels[i] + ':' + str(auc) + '\n')

    plt.xlabel('Ratio of unconsidered image [%]', fontsize=fontsize)
    plt.xlim([0, 98])
    plt.xticks(np.arange(0, 98, 10), fontsize=fontsize)
    plt.title(f"Testing on {db}, FMR={fmr}" + f" ({model})", fontsize=fontsize) # update : -3
        plt.ylabel('FNMR', fontsize=fontsize)

    plt.yticks(fontsize=fontsize)
    plt.grid(alpha=0.2)

    axbox = ax1.get_position()  #
    plt.legend(bbox_to_anchor=(axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.22), prop=FontProperties(size=12),
               loc='lower center', ncol=6)  #
    #ax1.get_legend().remove()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', db, db + '_' + str(fmr) +'_'+model + '.pdf'), bbox_inches='tight')

def perform_1v1_quality_eval(args):
 d = ['IJBC']
 d = args.eval_db.split(',')

 for dataset in d:
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-2)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir, dataset, str(1e-2)+"_auc.txt"))
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-3)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir, dataset, str(1e-3)+"_auc.txt"))
    if os.path.exists(os.path.join(args.output_dir, dataset, str(1e-4)+"_auc.txt")):
        os.remove(os.path.join(args.output_dir,  dataset, str(1e-4)+"_auc.txt"))

 match = True
 models=args.models.split(',')
 for model in models:
  for dataset in d:
    # create empty list for saving results at fnmr=1e-2, fnmr=1e-3, fnmr=1e-4,
    fnmrs_list_2, fnmrs_list_3, fnmrs_list_4, method_labels = [], [], []
    method_labels=[]
    method_names = args.method_name.split(',')

    if (not os.path.isdir(os.path.join(args.output_dir, dataset, 'fnmr'))):
        os.makedirs(os.path.join(args.output_dir, dataset, 'fnmr'))

    # 1. load all features based on number of images
    all_features = load_all_features(root=os.path.join(args.embeddings_dir, f"{dataset}_{model}"))
    unconsidered_rates = np.arange(0, 0.98, 0.05)
    desc = True
    for method_name in method_names:
        print(f"----process {model} {dataset} {method_name}-----------")
        targets, qlts = load_ijbc_pairs_quality(os.path.join(args.quality_score_dir, f"{method_name}_{dataset}.txt"))

        desc = False if method_name == 'PFE'

        if (desc):
            qlts_sorted_idx = np.argsort(qlts)  # [::-1]
        else:
            qlts_sorted_idx = np.argsort(qlts)[::-1]

        num_pairs = len(targets)
        fnmrs_list_2_inner = []
        fnmrs_list_3_inner = []
        fnmrs_list_4_inner = []

        for u_rate in tqdm(unconsidered_rates):
            # compute the used paris based on unconsidered rates
            hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs):]

            # load features based on hq_pairs_idx
            x, y = load_ijbc_pairs_features(os.path.join(args.quality_score_dir, f"{method_name}_{dataset}.txt"), all_features, hq_pairs_idx, args.feat_size)

            print('Calculate distance....')
            if args.distance_metric == 'cosine':
                dot = np.sum(np.multiply(x, y), axis=1)
                norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
                similarity = np.clip(dot/norm, -1., 1.)
                dist = np.arccos(similarity) / math.pi
                del dot, norm, similarity, x, y
                gc.collect()
            else:
                x = sklearn.preprocessing.normalize(x)
                y = sklearn.preprocessing.normalize(y)
                diff = np.subtract(x, y)
                dist = np.sum(np.square(diff), 1)
                del diff, x, y
                gc.collect()

            # sort in a desending order
            pos_dists =np.sort(dist[targets[hq_pairs_idx] == 1])
            neg_dists = np.sort(dist[targets[hq_pairs_idx] == 0])
            print('Compute threshold......')
            fmr100_th, fmr1000_th, fmr10000_th = get_eer_threshold(pos_dists, neg_dists, ds_scores=True)

            g_true = [g for g in pos_dists if g < fmr100_th]
            fnmrs_list_2_inner.append(1- len(g_true)/(len(pos_dists)))
            g_true = [g for g in pos_dists if g < fmr1000_th]
            fnmrs_list_3_inner.append(1 - len(g_true) / (len(pos_dists)))
            g_true = [g for g in pos_dists if g < fmr10000_th]
            fnmrs_list_4_inner.append(1 - len(g_true) / (len(pos_dists)))
            del pos_dists, neg_dists
            gc.collect()

        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr2.npy"), fnmrs_list_2_inner)
        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr3.npy"), fnmrs_list_3_inner)
        np.save(os.path.join(args.output_dir, dataset, 'fnmr', f"{method_name}_{model}_{dataset}_fnmr4.npy"), fnmrs_list_4_inner)

        fnmrs_list_2.append(fnmrs_list_2_inner)
        fnmrs_list_3.append(fnmrs_list_3_inner)
        fnmrs_list_4.append(fnmrs_list_4_inner)
        method_labels.append(f"{method_name}")

    save_pdf(fnmrs_list_2, method_labels, model=model, output_dir=args.output_dir, fmr=1e-2, db=dataset)
    save_pdf(fnmrs_list_3, method_labels, model=model, output_dir=args.output_dir, fmr=1e-3, db=dataset)
    save_pdf(fnmrs_list_4, method_labels, model=model, output_dir=args.output_dir, fmr=1e-4, db=dataset)

def main():
    args = parser.parse_args()
    perform_1v1_quality_eval(args)

if __name__ == '__main__':
    main()
