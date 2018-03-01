# -*- coding: utf-8 -*-
__author__ = 'lufo'

# this code can get image's feature using CNN

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import common
from scipy.io import savemat
import cv2
import cv2.cv as cv
import ConfigParser

cf=ConfigParser.ConfigParser()
cf.read('config.ini')
# Make sure that caffe is on the python path:
caffe_root = cf.get("Parameters","caffe_root")
sys.path.insert(0, caffe_root + 'python')
import caffe


def caffe_fea_extr(net, img_list, fea_dim, img_h, img_w, batch_num):
    """
    extract and save images' feature using caffe
    :param net: caffe net
    :param img_list: every elem is an image's path
    :param fea_dim: dimension of feature extract by caffe
    :param batch_num: number of images in a batch
    :param img_h: height of origin image
    :param img_w: width of origin image
    :param result_fold: save features in this path
    :return: feature list
    """
    fea_all = np.zeros((len(img_list) + batch_num, fea_dim))
    batch_list = (img_list[i:i + batch_num] for i in range(0, len(img_list), batch_num))
    img_batch = np.zeros((batch_num, 1, img_h, img_w))
    for j, batch in enumerate(batch_list):
        for i, img_path in enumerate(batch):
            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_RGB2GRAY)
	    img = cv2.resize(img,(144, 144))
            img_batch[i, 0, :, :] = img[-101:-1,-101:-1]
        net.blobs['data'].data[...] = img_batch/128
        fea = net.forward()['drop5']
	print fea.shape
        fea_all[(j * batch_num):((j + 1) * batch_num), :] = fea[:, :, 0, 0]
        # for item in ((k, v.data.shape) for k, v in net.blobs.items()):
        #    print(item)
    fea_all = fea_all[:len(img_list)]
    return fea_all


def get_feature(img_list):
    # get caffe net
    caffe.set_device(0)
    caffe.set_mode_gpu()  # set gpu model
    net = net_ul = caffe.Net('../models/CASIA_test.prototxt',
                    '../models/dr_iter.caffemodel',
                    caffe.TEST)

    return caffe_fea_extr(net, img_list, 320, 100, 100, 64)


def save_lfw_feature(img_list,name, result_fold='../data/'):
    fea_all = get_feature(img_list)
    savemat(result_fold + name+'.mat', {name: fea_all})


if __name__ == '__main__':
    # get image list
    img_list = []
    lable_list=[]
    with open('../0924_align.txt') as fr:
        for image_path in fr.readlines():
            img_list.append('/home/sam/caffe/dataset/casia_align_gray_50w/'+image_path.split(' ')[0])
	    lable_list.append(image_path.split(' ')[1])
    save_lfw_feature(img_list,'qf_casia_dr')
    lable_all = np.zeros((len(lable_list), 1))
    for i in range(len(lable_list)):
	lable_all[i]=int(lable_list[i])
    savemat('../data/' + 'qf_casia_lable.mat', {'qf_casia_label': lable_all})
