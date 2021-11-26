# Evaluation

## Evaluation on IJB-C
Follow this instruction to reproduce the results on IJB-C.
1. Download [IJB-C](https://www.nist.gov/programs-projects/face-challenges)
2. Unzip IJB-C (Folder structure should look like this /data/IJB_release/IJBC/)
3. Set (in extract_IJB.py) path = "/data/IJB_release/IJBC" to your IJB-C data folder and outpath = "/data/quality_data" where you want to save the preprocessed data
4. Run python extract_IJB.py (it creates the output folder, aligns and saves the images in BGR format, and creates image_path_list.txt and pair_list.txt)
5. Run xx to estimate the quality scores  
   5.1 For CR-FIQA(L)  
   5.2 For CR-FIQA(S)
6. Run CUDA_VISIBLE_DEVICES=0 python extract_emb.py --model_path ./pretrained/ElasticFace --model_id 295672 --dataset_path ./data/quality_data/IJBC
#### Aggregate IJB-C Templates and Create new Pair List
7. Copy CR-FIQA(L) scores to CR-FIQA/feature_extraction/quality_scores/CRFIQAL_IJBC.txt
8. (rename /data/quality_embeddings/IJBC_ElasticFaceModel to /data/quality_embeddings/IJBC_ElasticFaceModel_raw if not already done by previous script)
9. Run python ijb_qs_pair_file.py --dataset_path /data/IJB_release/IJBC --q_modelname CRFIQAL (it aggregates templates to match the pair file, saves them at /data/quality_embeddings/IJBC_ElasticFaceModel/ and creates a pair file with quality scores for each aggregated template and saves it at /data/quality_data/IJBC)
#### Evaluation with Quality Scores
10. Create folder CR-FIQA/feature_extraction/results/
11. CUDA_VISIBLE_DEVICES=0 python eval_ijbc_qs.py --model_path ./pretrained/ElasticFace --model_id 295672 --image-path /data/IJB_release/IJBC
