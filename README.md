# CR-FIQA


## This is the official repository of the paper:
#### CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability 

### CR-FIQA model training 
Model training:
In the paper, we employ MS1MV2 as the training dataset for CR-FIQA(L) which can be downloaded from InsightFace (MS1M-ArcFace in DataZoo)

Download MS1MV2 dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution

We use CASIA-WebFace as the training dataset for CR-FIQA(S) which can be downloaded from InsightFace (CASIA in DataZoo)
Download CASIA dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution



Unzip the dataset and place it in the data folder


Intall the requirement from requirement.txt

pip install -r requirements.txt

All code are trained and tested using PyTorch 1.7.1
Details are under (Torch)[https://pytorch.org/get-started/locally/]


### For training CR-FIQA(L)
Set the following in the config.py
1. config.output to output dir 
2. config.network = "iresnet100"
3. config.dataset = "emoreIresNet" 

### For training CR-FIQA(S)
Set the following in the config.py
1. config.output to output dir 
2. config.network = "iresnet50"
3. config.dataset = "webface" 

Run ./run.sh

## Pretrained model

#### [CR-FIQA(L)](https://drive.google.com/drive/folders/1siy_3eQSBuIV6U6_9wgGtbZG2GMgVLMy?usp=sharing)


#### [CR-FIQA(S)](https://drive.google.com/drive/folders/13bE4LP303XA_IzL1YOgG5eN0c8efHU9h?usp=sharing)

### Evaluation 
Follow these steps to reproduce the results on XQLFW.
1. Download the [XQLFW](https://martlgap.github.io/xqlfw/pages/download.html) (please download xqlfw_aligned_112.zip)
2. Unzip XQLFW (Folder structure should look like this ./data/XQLFW/xqlfw_aligned_112/)
3. Download also xqlfw_pairs.txt to ./data/XQLFW/xqlfw_pairs.txt
4. Set (in feature_extraction/extract_xqlfw.py) path = "./data/XQLFW" to your XQLFW data folder and outpath = "./data/quality_data" where you want to save the preprocessed data
5. Run python extract_xqlfw.py (it creates the output folder, saves the images in BGR format, creates image_path_list.txt and pair_list.txt)
6. Run evaluation/getQualityScore.py to estimate the quality scores  
   6.1 CR-FIQA(L)  
        6.1.1 Download the pretrained model
        6.1.1 run: python3 evaluation/getQualityScorce.py --data_dir "./data/quality_data" --datasets "XQLFW" --model_path "path_to_pretrained_CF_FIQAL_model" --backbone "iresnet100" --model_id "181952" --score_file_name "CRFIQAL.txt"
        
   6.2 CR-FIQA(S)
        6.1.1 Download the pretrained model
        6.1.1 run: python3 evaluation/getQualityScorce.py --data_dir "./data/quality_data" --datasets "XQLFW" --model_path "path_to_pretrained_CF_FIQAL_model" --backbone "iresnet50" --model_id "32572" --score_file_name "CRFIQAS.txt"
        
     
The quality score of LFW, AgeDB-30, CFP-FP, CALFW, CPLFW can be produced by following these steps:
1. LFW, AgeDB-30, CFP-FP, CALFW, CPLFW are be included in the training dataset folder [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
2. Set (in extract_bin.py) path = "/data/faces_emore/lfw.bin" to your LFW bin file and outpath = "./data/quality_data" where you want to save the preprocessed data (subfolder will be created)
3. Run python extract_bin.py (it creates the output folder, saves the images in BGR format, creates image_path_list.txt and pair_list.txt)  
4. Run evaluation/getQualityScore.py to estimate the quality scores  
   4.1 CR-FIQA(L)  
        4.1.1 Download the pretrained model
        4.1.1 run: python3 evaluation/getQualityScorce.py --data_dir "./data/quality_data" --datasets "XQLFW" --model_path "path_to_pretrained_CF_FIQAL_model" --backbone "iresnet100" --model_id "181952" --score_file_name "CRFIQAL.txt"
        
   4.2 CR-FIQA(S)
        4.1.1 Download the pretrained model
        4.1.1 run: python3 evaluation/getQualityScorce.py --data_dir "./data/quality_data" --datasets "XQLFW" --model_path "path_to_pretrained_CF_FIQAL_model" --backbone "iresnet50" --model_id "32572" --score_file_name "CRFIQAS.txt"
        
     
##### Ploting ERC curves 
7. Download pretrained model e.g. [ElasticFace-Arc](https://github.com/fdbtrs/ElasticFace) 295672backbone.pth, [MagFac](https://github.com/IrvingMeng/MagFace), [CurricularFace](https://github.com/HuangYG123/CurricularFace) or [ArcFace](https://github.com/deepinsight/insightface)
8. Run CUDA_VISIBLE_DEVICES=0 python feature_extraction/extract_emb.py --model_path ./pretrained/ElasticFace --model_id 295672 --dataset_path "./data/quality_data/XQLFW" --modelname "ElasticFaceModel"
    8.1 Note: change the path to pretrained model and other arguments according to the evaluated model 
9. Run python3 ERC/erc.py (details in  ERC/README.md)


