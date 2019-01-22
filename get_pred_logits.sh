#!/usr/bin/env bash

#GPUS=0,1
GPUS=0,1,2,3,4,5,6,7

# dollarstreet
#ROOT=/private/home/changhan/projects/image_classification_fairness/data
#~/miniconda3_6/bin/python get_pred_logits.py --resnet_size=101 --data_format=NCHW --class_num=11166 \
#    --visiable_gpus=${GPUS} --flush_every 10000 --pretrain_ckpt=checkpoints/ckpt-resnet101-mlimages/model.ckpt \
#    --result=${ROOT}/dollarstreet_tencent_predictions.pkl --images=${ROOT}/dollarstreet_img_list.txt

# oix
ROOT=/private/home/changhan/projects/image_classification_fairness/data
~/miniconda3_6/bin/python get_pred_logits.py --resnet_size=101 --data_format=NCHW --class_num=11166 \
    --visiable_gpus=${GPUS} --flush_every 20000 --pretrain_ckpt=checkpoints/ckpt-resnet101-mlimages/model.ckpt \
    --result=${ROOT}/oix_tencent_predictions.pkl --images=${ROOT}/oix_img_list.txt
