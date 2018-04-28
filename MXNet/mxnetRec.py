# -*- coding: utf-8 -*-
import sys
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import time
import types

import face_embedding
import argparse
import cv2
import numpy as np
import cPickle as pickle
import xlwt
import shutil


def get_parser():
  parser = argparse.ArgumentParser(description='face model test')
  # general
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--model', default='models/model-r34-amf/model,0', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
  parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
  parser.add_argument('--threshold', default=1.09, type=float, help='ver dist threshold')

  parser.add_argument('--mtcnn', default=False, type=bool, help='whether use mtcnn, if not use dlib')
  parser.add_argument('--data_path', default='../starsPic620', type=str, help='the path of image base')
  parser.add_argument('--db_save_path', default='database/20180426_db600_20stars_hou.pickle', type=str, help='the save path of data base -- that is the feature of all images')
  args = parser.parse_args()
  return args

class FaceIdendify():
  def __init__(self, args):
    self.model = face_embedding.FaceModel(args)
    self.args = args
    self.database = {'Name': 'Feature'};
    self.database.clear();
    self.num = 1
    if not os.path.exists(self.args.db_save_path):
      self.create_database()
    else:
      print(self.args.db_save_path, 'exists. Loading...')
      f = open(self.args.db_save_path,'rb')
      self.database = pickle.load(f)
      f.close()

  def just_test(self):
    img = cv2.imread('/home/peterhou/vic/data/know/chuanpu.jpg')
    f1 = self.model.get_feature(img)[0]
    img = cv2.imread('/home/peterhou/vic/data/unknow/unknow.jpg')
    f2 = self.model.get_feature(img)[0]
    dist = np.sum(np.square(f1-f2))
    print(dist)
    sim = np.dot(f1, f2.T)
    print(sim)

  def create_database(self): 
    path = self.args.data_path
    files= os.listdir(path)
    cnt=0
    for file in files:
      if file.find('.')==-1:
          continue
      if(file.split('.')[1][0]=='p'):
          ppath=path+'/'+file.split('.')[0]+'.png'
      elif(file.split('.')[1][0]=='j'):
          ppath=path+'/'+file.split('.')[0]+'.jpg'
      elif(file.split('.')[1][0]=='J'):
          ppath=path+'/'+file.split('.')[0]+'.JPG'
      else:
        continue
      print file.split('.')[1][0]
      print ppath
      im=cv2.imread(ppath)
      #feat = self.model.get_feature(im)[0]
      feat,d,p = self.model.get_feature(im)
      self.database[file.split('.')[0]]=feat
      d = d[0]
      [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
      imgFace = im[y1:y2,x1:x2,:]
      cv2.imwrite('../DatabaseFace/'+file.split('.')[0]+'.jpg',imgFace)
      cnt=cnt+1
      print cnt

    f = open(self.args.db_save_path, 'wb')
    pickle.dump(self.database,f,-1)
    f.close()
    print cnt, 'images have been done'

  def _get_score(self, f1, f2):
    #sim = np.dot(source_feature, target_feature.T)
    diff = np.subtract(f1, f2)
    dist = np.sum(np.square(diff))
    return dist

  def getscore(self, feat):
    minscore= 1000.
    anskey='nobody'
    for key,value in self.database.items():
        score=self._get_score(feat, value)
        if(score<minscore):
            minscore=score
            anskey=key
    if(minscore > self.args.threshold):
        anskey='Stranger_'+anskey
        #anskey='Stranger'
    fileRes = r'D:/face_recognition/MXNet/videoout/20170528.txt'
    with open(fileRes,'a+') as fr:
      fr.write('No. '+ str(self.num) +' anskey = '+str(anskey)+'  minscore = '+str(minscore)+'\r')
    print ('No. '+ str(self.num) +' anskey = '+str(anskey)+'  minscore = '+str(minscore))
    self.num += 1
    return anskey,minscore

  def get_features(self, img):
    feats, dets, points = self.model.get_feature(img)
    return feats, dets, points


def drawrec(minkey,minscore,img,d,video_name):
#    colorbook={'Stranger':(0,0,0)}
#    if(minkey in colorbook.keys()):
#        color=colorbook[minkey]
  if('Stranger' in minkey):
    color = (0,0,0)
  else:
    color=(255,255,255)
  cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),color,2)
  fontsize=(d.right()-d.left())/200.
  sidebar=int(round((d.right()-d.left())/15.))
  font = cv2.FONT_HERSHEY_COMPLEX
  cv2.putText(img,minkey,(d.left()+sidebar,d.top()-sidebar*5),font,1,color,1,False)
  cv2.putText(img,'%2f'%(minscore),(d.left()+sidebar,d.top()-sidebar*1),font,1,color,1,False)


def det2face(img, d):
  [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
  if x1<0:
      x1 = 10
  if y1<0:
      y1 = 10
  return img[y1:y2,x1:x2,:]

def mxnetFaceRec(img, face_identify):
  resultList = []
  feats, dets, points_all = face_identify.get_features(img)
  if(not feats):
    print "No face detected"
    minkey = "Stranger"
    minscore = 100
    imgFace = img
    resultList.append([minkey,minscore,imgFace])
  else:
    for i, feat in enumerate(feats):
      minkey,minscore = face_identify.getscore(feat)
      imgFace = det2face(img, dets[i])
      resultList.append([minkey,minscore,imgFace])
  return resultList


def mxnetCompare(img1,img2,face_identify):
  feats1, dets1, points_all1 = face_identify.get_features(img1)
  feats2, dets2, points_all2 = face_identify.get_features(img2)
  score = 100
  imgFace = img1
  '''
  if(not feats1):
    score = 100
    imgFace = img1
  '''
  if (feats1):
    print ("feats1 lenth is {}".format(len(feats1)))
    for feat1 in feats1:
      s = np.sum(np.square(np.subtract(feat1, feats2)))
      if s<score:
        score = s
        imgFace = det2face(img1, dets1[0])
  return score, imgFace

def videoFile2videoFile():
  args = get_parser()
  face_identify = FaceIdendify(args)
  #testvideo_path = '/home/peterhou/FaceRecognition/Testdata1/'
  testvideo_path = 'D:/face_recognition/LinClassFaces_allvideo/'
  files= os.listdir(testvideo_path)
  #anspath='/home/peterhou/vic/data/videos/myout/'
  anspath ='D:/face_recognition/LinClassFaces_allvideo_result/'
  fourcc = cv2.VideoWriter_fourcc(*"DIVX")
  #fileRes = r'20170528.txt'
  #fileRes = r'D:/face_recognition/MXNet/videoout/20170528.txt'

  for filename in files:
    if filename.split('.')[-1] not in ['avi', 'mp4']:
      continue
    try:
      capture=cv2.VideoCapture(testvideo_path+filename)
    except:
      continue
    out = cv2.VideoWriter(anspath+filename, fourcc, 10.0, (1024,768))
    #fileRes = r'20170528.txt'
    #with open(fileRes,'a+') as fr:
    #  print ("writing !!!!!!!!!!!!!!!!!!")
    #  fr.write(filename +'\r')
    #face_identify.num = 1 
    while capture.read():
      for i in range(20):
        ret,img=capture.read()
      if isinstance(img, types.NoneType):
        print ('Not a image. This is the end of the vedio.')
        break #This is the end of the vedio.
      #continue
      else:
        print img.shape
        feats, dets, points_all = face_identify.get_features(img)
        if(not feats):
          print "No face detected"
          img=cv2.resize(img,(1024,768))
          cv2.imshow('Face Verification',img)
          cv2.waitKey(10)
          out.write(img)
          continue
        for i, feat in enumerate(feats):
          minkey,minscore = face_identify.getscore(feat)
          #fileRes = r'20170528.txt'
          #with open(fileRes,'a+') as fr:
          #  fr.write('No. '+ str(face_identify.num-1) +' anskey = '+str(minkey)+'  minscore = '+str(minscore)+'\r')
          d = dets[i]
          drawrec(minkey,minscore,img,d,filename.split('.')[0])
        img=cv2.resize(img,(1024,768))
        cv2.imshow('Face Verification',img)
        cv2.waitKey(10)
        out.write(img)
    capture.release()
    out.release()
  cv2.destroyAllWindows()



def strangerFAR():  # FAR: False Acceptance Rate; FRR: False Rejection Rate
  # e.g.: 
  # 2000 different faces
  # 500(in a file) as database
  # 1500(in a file) as input
  # error occurs when the result is NOT "Stranger"

  args = get_parser()
  face_identify = FaceIdendify(args)
  inputPath = "D:/face_recognition/Database1600/1000_input/"
  errorPicPath = "D:/face_recognition/Database1600/errorPair/"
  fileRes = r'D:/face_recognition/MXNet/1000_input.txt'
  excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
  cnt = 1
  faces = os.listdir(inputPath)
  for face in faces:
    facePath = inputPath+face
    img = cv2.imread(facePath)
    feats, dets, points_all = face_identify.get_features(img)
    for i, feat in enumerate(feats):
          minkey,minscore = face_identify.getscore(feat)
          break # one face default
    with open(fileRes,'a+') as fr:
      fr.write('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore)+'\r')

    #print ('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore))
    sheet.write(cnt-1, 0, 'cnt')
    sheet.write(cnt-1, 1, cnt)
    sheet.write(cnt-1, 2, 'minkey')
    sheet.write(cnt-1, 3, minkey)
    sheet.write(cnt-1, 4, 'score')
    sheet.write(cnt-1, 5, minscore.item())
    if minscore<1.09:
      shutil.copyfile(facePath, errorPicPath+minkey+face)
      shutil.copyfile('D:/face_recognition/DatabaseFace/'+minkey+'.jpg', errorPicPath+minkey+'.jpg')
    print cnt
    cnt+=1
  excel.save(r'D:/face_recognition/MXNet/score1000.xls')

def familiarFRR():  # FAR: False Acceptance Rate; FRR: False Rejection Rate
  # e.g.: 
  # 500 faces (in a file) as database
  # 40 of them have 2000 different faces as input
  # error occurs when the result is NOT the right answer

  args = get_parser()
  face_identify = FaceIdendify(args)
  inputPath = "D:/face_recognition/starsVideoFaces/"
  #errorPicPath = "D:/face_recognition/Database1600/errorPair/"
  labelPath = "D:/face_recognition/starsVideoFaces_label/"
  fileRes = r'D:/face_recognition/MXNet/stars_video_faces_mxnet.txt'
  excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
  cnt = 1
  faces = os.listdir(inputPath)
  for face in faces:
    facePath = inputPath+face
    img = cv2.imread(facePath)
    feats, dets, points_all = face_identify.get_features(img)
    if(not feats):
      minkey = "Stranger_no_face"
      minscore = 111*np.ones(1)
    else:
      for i, feat in enumerate(feats):
            minkey,minscore = face_identify.getscore(feat)
            break # one face default
    with open(fileRes,'a+') as fr:
      fr.write('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore)+'\r')

    #print ('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore))
    sheet.write(cnt-1, 0, 'cnt')
    sheet.write(cnt-1, 1, cnt)
    sheet.write(cnt-1, 2, 'minkey')
    sheet.write(cnt-1, 3, minkey)
    sheet.write(cnt-1, 4, 'score')
    sheet.write(cnt-1, 5, minscore.item())
    shutil.copyfile(facePath, labelPath+minkey+face)
    print cnt
    cnt+=1
  excel.save(r'D:/face_recognition/MXNet/stars_video_faces_mxnet.xls')


if __name__ == '__main__':
  videoFile2videoFile()
  print ('done!')







