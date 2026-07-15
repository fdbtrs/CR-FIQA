import argparse
import os
import sys

import numpy as np
from tqdm import tqdm


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./data/quality_data/XQLFW',
                        help='dataset directory')
    parser.add_argument('--modelname', type=str, default='ElasticFaceModel',
                        help='ArcFaceModel, CurricularFaceModel, ElasticFaceModel, MagFaceModel')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="",
                        help='path to pretrained model.')
    parser.add_argument('--model_id', type=str, default="295672",
                        help='digit number in backbone file name')
    parser.add_argument('--relative_dir', type=str, default='./data',
                        help='path to save the embeddings')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')
    
    # --- NEW ARGUMENTS ---
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction to prevent GPU OOM.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip image paths where the embedding file already exists.')
    
    return parser.parse_args(argv)


def read_img_path_list(image_path_file, relative_dir):
    with open(image_path_file, "r") as f:
        lines = f.readlines()
        lines = [os.path.join(relative_dir, line.rstrip()) for line in lines]
    return lines


def main(param):
    dataset_path = param.dataset_path
    modelname = param.modelname
    gpu_id = param.gpu_id
    data_path = param.relative_dir
    dataset_name = dataset_path.split("/")[-1]
    
    out_path = os.path.join(data_path, "quality_embeddings", f"{dataset_name}_{modelname}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    image_path_list = read_img_path_list(os.path.join(dataset_path, "image_path_list.txt"), data_path)
    
    # --- FEATURE: Resume / Skip ---
    if param.resume:
        filtered_paths = []
        for path in image_path_list:
            filename = str(path.split("/")[-1].split(".")[0])
            out_file = os.path.join(out_path, f"{filename}.npy")
            if not os.path.exists(out_file):
                filtered_paths.append(path)
        print(f"Resume Mode: Skipping {len(image_path_list) - len(filtered_paths)} already processed files.")
        image_path_list = filtered_paths

    if not image_path_list:
        print("All embeddings already exist. Nothing to extract!")
        return

    # Load chosen model
    if modelname == "ArcFaceModel":
        from model.ArcFaceModel import ArcFaceModel
        face_model = ArcFaceModel(param.model_path, param.model_id, gpu_id=gpu_id)
    elif modelname == "CurricularFaceModel":
        from model.CurricularFaceModel import CurricularFaceModel
        face_model = CurricularFaceModel(param.model_path, param.model_id, gpu_id)
    elif modelname == "ElasticFaceModel":
        from model.ElasticFaceModel import ElasticFaceModel
        face_model = ElasticFaceModel(param.model_path, param.model_id, gpu_id)
    elif modelname == "MagFaceModel":
        from model.MagFaceModel import MagFaceModel
        face_model = MagFaceModel(param.model_path, param.model_id, gpu_id)
    else:
        print("Unknown model")
        sys.exit()

    # --- FEATURE: Batching & Inference ---
    print(f"Extracting embeddings in batches of {param.batch_size}...")
    
    for i in tqdm(range(0, len(image_path_list), param.batch_size), desc="Extraction Batches"):
        batch_paths = image_path_list[i : i + param.batch_size]
        
        # Extract features for current batch
        features = face_model.get_batch_feature(batch_paths, flip=0, color=param.color_channel)
        features_flipped = face_model.get_batch_feature(batch_paths, flip=1, color=param.color_channel)
        
        # Concatenate and save each item in the batch
        for j, path in enumerate(batch_paths):
            conc_features = np.concatenate((features[j], features_flipped[j]), axis=0)
            filename = str(path.split("/")[-1].split(".")[0])
            np.save(os.path.join(out_path, filename), conc_features)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
