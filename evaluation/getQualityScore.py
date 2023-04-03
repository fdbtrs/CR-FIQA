import argparse
import os
import sys

from evaluation.QualityModel import QualityModel


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Root dir for evaluation dataset')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--datasets', type=str, default='XQLFW',
                        help='list of evaluation datasets (,)  e.g.  XQLFW, lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="/home/fboutros/LearnableMargin/output/ResNet50-COSQSArcFace_SmothL1",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="32572",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="iresnet50",
                        help=' iresnet100 or iresnet50 ')
    parser.add_argument('--score_file_name', type=str, default="quality_r50.txt",
                        help='score file name, the file will be store in the same data dir')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')
    #

    return parser.parse_args(argv)

def read_image_list(image_list_file, image_dir=''):
    image_lists = []
    with open(image_list_file) as f:
        absolute_list=f.readlines()
        for l in absolute_list:
            image_lists.append(os.path.join(image_dir, l.rstrip()))
    return image_lists, absolute_list
def main(param):
    datasets=param.datasets.split(',')
    face_model=QualityModel(param.model_path,param.model_id, param.gpu_id, param.backbone)
    for dataset in datasets:
        root=os.path.join(param.data_dir)
        image_list, absolute_list=read_image_list(os.path.join(param.data_dir,'quality_data',dataset,'image_path_list.txt'), root)
        embedding, quality=face_model.get_batch_feature(image_list,batch_size=16, color=param.color_channel)
        if (os.path.isdir(os.path.join(param.data_dir,'quality_data',dataset))):
            os.makedirs(os.path.join(param.data_dir,'quality_data',dataset))
        quality_score=open(os.path.join(param.data_dir,'quality_data',dataset,param.score_file_name),"a")
        for i in range(len(quality)):
            quality_score.write(absolute_list[i].rstrip()+ " "+str(quality[i][0])+ "\n")

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))