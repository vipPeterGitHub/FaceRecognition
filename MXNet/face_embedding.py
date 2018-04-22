from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess

import dlib


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, args):
    self.args = args
    model = edict()

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.4,0.6,0.6]
    self.det_factor = 0.9
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.image_size = image_size
    _vec = args.model.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    ctx = mx.gpu(args.gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    if args.mtcnn:
      mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=mx.cpu(), num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
      predictor = None
    else:
      predictor_path = 'dlib-model/shape_predictor_68_face_landmarks.dat'
      detector = dlib.get_frontal_face_detector()
      predictor = dlib.shape_predictor(predictor_path)
    self.detector = detector
    self.predictor = predictor

  def get_face_dets_points(self, face_img):
    if self.predictor:
      dets = self.detector(face_img)
      if len(dets)==0:
        return None, None
      points_all = []
      for j, bbox in enumerate(dets):
        shape = self.predictor(face_img, bbox)
        sevens=np.zeros((9,2))
        index=[30,39,36,42,45,48,54,37,43] # nt,lei,leo,rei,reo,lm,rm
        for i in range(len(index)):
          cord=str(shape.part(index[i])).split(',')
          x=int(cord[0][1:])
          y=int(cord[1][:-1])
          sevens[i][0]=x
          sevens[i][1]=y
        points = np.array([(sevens[1]+sevens[2])/2,(sevens[3]+sevens[4])/2,sevens[0],sevens[5],sevens[6]])
        points_all.append(points)
    else:
#      ret = self.detector.detect_face_limited(face_img, det_type = self.args.det)
      ret = self.detector.detect_face(face_img)
      if ret is None:
        return None, None
      dets, points_all = ret
      if bbox.shape[0]==0:
        return None, None
      for i, bbox in enumerate(dets):
        dets[i] = dets[i,0:4]
        points_all = points_all[i,:].reshape((2,5)).T
    return dets, points_all


  def get_feature(self, face_img):
    #face_img is bgr image
    dets, points_all = self.get_face_dets_points(face_img)
    if not dets:
      return None, None, None
 #   print('bbox:',bbox)
 #   print('points',points)
    embedding_all = []
    for i, bbox in enumerate(dets):
      points = points_all[i]
      nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      #print(nimg.shape)
      embedding = None
      for flipid in [0,1]:
        if flipid==1:
          if self.args.flip==0:
            break
          do_flip(aligned)
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        _embedding = self.model.get_outputs()[0].asnumpy()
        #print(_embedding.shape)
        if embedding is None:
          embedding = _embedding
        else:
          embedding += _embedding
      embedding = sklearn.preprocessing.normalize(embedding).flatten()
      embedding_all.append(embedding)
    return embedding_all, dets, points_all


