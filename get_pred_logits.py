#!/usr/bin/python

"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

# Use pre-trained model extract image feature

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 as cv
import tensorflow as tf
from models import resnet as resnet
from flags import FLAGS
from tqdm import tqdm
import pickle as pkl
import os

tf.app.flags.DEFINE_string("result", "", "file name to save features")
tf.app.flags.DEFINE_string("images", "", "contains (image path, image id) per line per image")
tf.app.flags.DEFINE_float("prob_thres", 0.5, "filter logit output by threshold")

"""
Crop Image To 224*224
 Args:
   img: an 3-D numpy array (H,W,C)
   type: crop method support [ center | 10crop ]
"""


def preprocess(img, cropping_type="center"):
    # resize image with smallest side to be 256
    rawH = float(img.shape[0])
    rawW = float(img.shape[1])
    newH = 256.0
    newW = 256.0
    if rawH <= rawW:
        newW = (rawW/rawH) * newH
    else:
        newH = (rawH/rawW) * newW
    img = cv.resize(img, (int(newW), int(newH)))
    if cropping_type == 'center':
        imgs = np.zeros((1, 224, 224, 3))
        imgs[0, ...] = img[int((newH-224)/2):int((newH-224)/2)+224, int((newW-224)/2):int((newW-224)/2)+224]
    elif cropping_type == '10crop':
        imgs = np.zeros((10, 224, 224, 3))
        offset = [(0, 0),
                  (0, int(newW-224)),
                  (int(newH-224), 0),
                  (int(newH-224), int(newW-224)),
                  (int((newH-224)/2), int((newW-224)/2))]
        for i in range(0, 5):
            imgs[i, ...] = img[offset[i][0]:offset[i][0]+224, offset[i][1]:offset[i][1]+224]
        img = cv.flip(img, 1)
        for i in range(0, 5):
            imgs[i+5, ...] = img[offset[i][0]:offset[i][0]+224, offset[i][1]:offset[i][1]+224]
    else:
        raise ValueError("Type not support")
    imgs = ((imgs/255.0) - 0.5) * 2.0
    imgs = imgs[..., ::-1]
    return imgs


def main():
    # build model
    images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
    net = resnet.ResNet(images, is_training=False)
    net.build_model()
    logits = net.logit
    # restore model
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = FLAGS.visiable_gpus
    config.log_device_placement=False
    sess = tf.Session(config=config)
    # load trained model
    saver.restore(sess, FLAGS.pretrain_ckpt)
    # inference
    types = 'center'

    with open(FLAGS.images) as f:
        img_paths = [l.strip().split('\t') for l in f]

    cache = {}
    if os.path.exists(FLAGS.result):
        with open(FLAGS.result, 'rb') as f:
            cache = pkl.load(f)

    unfinished = [(p, i) for p, i in img_paths if i not in cache]
    progress = tqdm(unfinished)
    n_errors = 0
    for img_path, img_id in progress:
        raw_img = cv.imread(img_path)
        if raw_img is None or raw_img.data is None:
            n_errors += 1
        imgs = preprocess(raw_img, types)
        output = sess.run(logits, {images: imgs})
        output = np.squeeze(output[0])
        if types == '10crop':
            output = np.mean(output, axis=0)
        idx = output.argsort()[::-1]
        output = idx[output[idx] > FLAGS.prob_thres].tolist()
        cache[img_id] = output
        progress.set_description("n_errors: {}".format(n_errors))
        if len(cache) % FLAGS.flush_every == 0:
            with open(FLAGS.result, 'wb') as f:
                pkl.dump(cache, f, protocol=4)

    with open(FLAGS.result, 'wb') as f:
        pkl.dump(cache, f, protocol=4)


if __name__ == '__main__':
    main()
