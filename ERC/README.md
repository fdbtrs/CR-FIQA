# ERC plot
---

## Data preparation
For plotting Error vs. Reject Curve (ERC) and using the following code, the format of the directories and files are:
1. Embedding directory that save embeddings computed from different FR models:
```
+-- embeddings_dir
|   +-- dataset_FRmodel   #e.g., lfw_ArcFaceModel
|       +-- xxx.npy
```
2. #### Quality score directory that save image pair list and quality scores
```
+-- quality_score_dir
|   +-- dataset   # e.g., lfw
|       +-- images  # where save the face images
|       +-- pair_list.txt
|       +-- qualityEstimationMethod_dataset.txt  #e.g., CRFIQAL_lfw.txt
```
where the format of pair_list.txt is:
```
image_name_1, image_name_2, type
# 1: genuine, 0: Imposter
00000 00001 1  
00002 00003 1
05402 05405 0
05404 05409 0
```
and the format of qualityEstimationMethod_dataset.txt is:
```
image_path, quality score
quality_score_dir/lfw/images/00000.jpg 0.36241477727890015
quality_score_dir/lfw/images/00000.jpg 0.36241477727890015
quality_score_dir/lfw/images/00001.jpg 0.37981975078582764
quality_score_dir/lfw/images/00002.jpg 0.44192782044410706
quality_score_dir/lfw/images/00003.jpg 0.5173501372337341
```

## ERC evaluation
1. After preparing the above data, erc.py can be used for compute and plot:
    ```
    python erc.py \
      --embeddings_dir 'embeddings_dir' \
      --quality_score_dir 'quality_score_dir' \
      --method_name 'CRFIQAL,CRFIQAS,FaceQnet,DeepIQA,PFE,..' \
      --models 'ArcFaceModel, ElasticFaceModel, MagFaceModel, CurricularFaceModel' \
      --eval_db 'adience,XQLFW,lfw,agedb_30,calfw,cplfw,cfp_fp' \
      --distance_metric 'cosine' \
      --output_dir 'output_dir' \
    ```
    More details can be checked in erc.py

---

## Data preparation for IJB-C
Becuase the IJB-C dataset is a large-scale dataset, the way for loading embeddings and quality scores is a little bit different due to the limited computation resource.
For plotting Error vs. Reject Curve (ERC) and using the following code, the format of the directories and files are:
1. Embedding directory that save embeddings computed from different FR models:
```
+-- embeddings_dir
|   +-- dataset_FRmodel   #e.g., IJBC_ArcFaceModel
|       +-- xxx.npy
```
2. Quality score directory that save image pair list and quality scores
```
+-- quality_score_dir
|   +-- dataset   # e.g., lfw
|       +-- images  # where save the face images
|       +-- qualityEstimationMethod_dataset.txt  #e.g., CRFIQAL_IJBC.txt
```
where the format of qualityEstimationMethod_dataset.txt is:
```
image_name_1, image_name_2, type, quality score # in type column: 1: genuine, 0: Imposter
171707  187569  0  1.7933012
26      17991   0  1.764878
878     20568   1  0.5316349
170001  178303  1  0.10600686
```

## ERC evaluation for IJB-C
1. After preparing the above data, erc_ijbc.py can be used for compute and plot:
    ```
    python erc.py \
      --embeddings_dir 'embeddings_dir' \
      --quality_score_dir 'quality_score_dir' \
      --method_name 'CRFIQAL,CRFIQAS,FaceQnet,DeepIQA,PFE,..' \
      --models 'ArcFaceModel, ElasticFaceModel, MagFaceModel, CurricularFaceModel' \
      --eval_db 'IJBC' \
      --distance_metric 'cosine' \
      --output_dir 'output_dir' \
    ```
    More details can be checked in erc_ijbc.py
