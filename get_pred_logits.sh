#!/usr/bin/env bash

GPUS=0
#GPUS=0,1,2,3,4,5,6,7

export CUDA_VISIBLE_DEVICES=${GPUS}

ROOT=/private/home/changhan/projects/image_classification_fairness/data

PY=~/miniconda3_6/bin/python

# dollarstreet
#${PY} get_pred_logits.py --resnet_size=101 --data_format=NCHW --class_num=11166 \
#    --visiable_gpus=${GPUS} --flush_every 5000 --pretrain_ckpt=checkpoints/ckpt-resnet101-mlimages/model.ckpt \
#    --result=${ROOT}/tencent_dollarstreet_predictions.pkl --images=${ROOT}/dollarstreet_img_list.txt --prob_thres 0.5

# openimages val
#${PY} get_pred_logits.py --resnet_size=101 --data_format=NCHW --class_num=11166 \
#    --visiable_gpus=${GPUS} --flush_every 5000 --pretrain_ckpt=checkpoints/ckpt-resnet101-mlimages/model.ckpt \
#    --result=${ROOT}/tencent_oi_val_predictions.pkl --images=${ROOT}/openimages_val_img_list.txt --prob_thres 0.5


# oix
${PY} get_pred_logits.py --resnet_size=101 --data_format=NCHW --class_num=11166 \
    --visiable_gpus=${GPUS} --flush_every 5000 --pretrain_ckpt=checkpoints/ckpt-resnet101-mlimages/model.ckpt \
    --result=${ROOT}/tencent_oix_predictions.pkl --images=${ROOT}/oix_img_list.txt --prob_thres 0.5
