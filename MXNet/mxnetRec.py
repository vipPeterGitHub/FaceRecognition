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
  parser.add_argument('--data_path', default='../DBczm', type=str, help='the path of image base')
  parser.add_argument('--db_save_path', default='database/20180426_db600_20stars_beijing&GOT_hou.pickle', type=str, help='the save path of data base -- that is the feature of all images')
  #parser.add_argument('--db_save_path', default='database/DB_czm.pickle', type=str, help='the save path of data base -- that is the feature of all images')
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
    img = cv2.imread('D:/face_recognition/lfw_new/Aaron_Peirsol_0001.jpg')
    f1 = self.model.get_feature(img)[0]
    #print(f1)
    img = cv2.imread('D:/face_recognition/lfw_new/Aaron_Peirsol_0003.jpg')
    f2 = self.model.get_feature(img)[0]
    #print(f2)
    dist = np.sum(np.square(np.subtract(f1,f2)))
    print(dist)
    #dist = np.sum(np.square(f1[0]-f2[0]))
    #print(dist)
    #print(len(f1))
    sim = np.dot(f1[0], f2[0].T)
    print(sim)

  def test_one_vs_more(self):
    fileRes = r'D:/face_recognition/MXNet/one_vs_more_mxnet.txt'
    img = cv2.imread('D:/face_recognition/lfw_new/Aaron_Peirsol_0003.jpg')
    f1 = self.model.get_feature(img)[0]
    rootPath = 'D:/face_recognition/lfw_new/'
    cnt = 0
    names = os.listdir(rootPath)
    for name in names:
      try:
        facePath = rootPath+name
        img = cv2.imread(facePath)
        f2 = self.model.get_feature(img)[0]
        dist = np.sum(np.square(np.subtract(f1,f2)))
        #print(dist)
        sim = np.dot(f1[0], f2[0].T)
        #print(sim)
        cnt += 1
        with open(fileRes,'a+') as fr:
          fr.write('No. '+ str(cnt) +' dist = '+str(dist)+'  sim = '+str(sim)+'\r')
        print('No. '+ str(cnt) +' dist = '+str(dist)+'  sim = '+str(sim)+'\r')
        if cnt == 200:
          break
      except:
        pass

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
    fileRes = r'D:/face_recognition/MXNet/LFW_FAR.txt'
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
  testvideo_path = 'D:/face_recognition/tmpVideoIn/'
  files= os.listdir(testvideo_path)
  #anspath='/home/peterhou/vic/data/videos/myout/'
  anspath ='D:/face_recognition/tmpVideoOut/'
  fourcc = cv2.VideoWriter_fourcc(*"DIVX")
  #fileRes = r'20170528.txt'
  #fileRes = r'D:/face_recognition/MXNet/videoout/20170528.txt'
  excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)

  for filename in files:
    if filename.split('.')[-1] not in ['avi', 'mp4', 'mov']:
      continue
    try:
      capture=cv2.VideoCapture(testvideo_path+filename)
    except:
      continue
    out = cv2.VideoWriter(anspath+filename, fourcc, 20.0, (1024,768))
    #fileRes = r'20170528.txt'
    #with open(fileRes,'a+') as fr:
    #  print ("writing !!!!!!!!!!!!!!!!!!")
    #  fr.write(filename +'\r')
    #face_identify.num = 1 
    cnt = 1
    while capture.read():
      for i in range(1):
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

          sheet.write(cnt-1, 0, 'cnt')
          sheet.write(cnt-1, 1, cnt)
          sheet.write(cnt-1, 2, 'minkey')
          sheet.write(cnt-1, 3, minkey)
          sheet.write(cnt-1, 4, 'score')
          sheet.write(cnt-1, 5, minscore.item())
          print cnt
          cnt += 1
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
    excel.save(r'D:/face_recognition/MXNet/20170528excel.xls')
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
  inputPath = "D:/face_recognition/testFile656/beijing_GOT_faces/"
  #errorPicPath = "D:/face_recognition/Database1600/errorPair/"
  labelPath = "D:/face_recognition/testFile656/beijing_GOT_faces_label/"
  fileRes = r'D:/face_recognition/testFile656/beijing_GOT_faces_mxnet.txt'
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
  excel.save(r'D:/face_recognition/testFile656/beijing_GOT_faces_mxnet.xls')




def LFW_FAR():
  args = get_parser()
  face_identify = FaceIdendify(args)
  #testvideo_path = '/home/peterhou/FaceRecognition/Testdata1/'
  LFW_path = 'D:/face_recognition/lfw_new/'
  pics= os.listdir(LFW_path)
  fileRes = r'lfw_result.txt'
  #fileRes = r'D:/face_recognition/MXNet/videoout/20170528.txt'
  excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
  cnt = 1
  for pic in pics:
    face =  LFW_path + pic
    img = cv2.imread(face)
    feats, dets, points_all = face_identify.get_features(img)
    if(not feats):
      continue
    for i, feat in enumerate(feats):
      minkey,minscore = face_identify.getscore(feat)
      break
    sheet.write(cnt-1, 0, 'cnt')
    sheet.write(cnt-1, 1, cnt)
    sheet.write(cnt-1, 2, 'minkey')
    sheet.write(cnt-1, 3, minkey)
    sheet.write(cnt-1, 4, 'score')
    sheet.write(cnt-1, 5, minscore.item())
    cnt += 1
  excel.save(r'D:/face_recognition/MXNet/LFW_FAR_excel.xls')

def LFW_FRR():
  args = get_parser()
  face_identify = FaceIdendify(args)
  excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
  DBPath = 'D:/face_recognition/lfw/'
  cnt = 1
  #history_path = 'D:/testImage/history/'
  names = os.listdir(DBPath)
  for name in names:
    facesPath = DBPath+name
    faces = os.listdir(facesPath)
    #resPath = facesPath+'/'+faces[0]
    if len(faces) > 1:
      #print len(faces)
      flag = 0
      for face in faces:
        resPath = facesPath+'/'+face
        img = cv2.imread(resPath)
        feats, dets, points_all = face_identify.get_features(img)
        if flag == 0:

          if (not feats):
            sheet.write(cnt-1, 0, 'cnt')
            sheet.write(cnt-1, 1, cnt)
            sheet.write(cnt-1, 2, 'name')
            sheet.write(cnt-1, 3, face)
            sheet.write(cnt-1, 4, 'score')
            sheet.write(cnt-1, 5, 999)
            print cnt
            cnt += 1
            continue

          f0 = feats
          flag = 1
          score = 0

          sheet.write(cnt-1, 0, 'cnt')
          sheet.write(cnt-1, 1, cnt)
          sheet.write(cnt-1, 2, 'name')
          sheet.write(cnt-1, 3, face)
          sheet.write(cnt-1, 4, 'score')
          sheet.write(cnt-1, 5, score)
          print cnt
          cnt += 1
        else:
          if (not feats):
            sheet.write(cnt-1, 0, 'cnt')
            sheet.write(cnt-1, 1, cnt)
            sheet.write(cnt-1, 2, 'name')
            sheet.write(cnt-1, 3, face)
            sheet.write(cnt-1, 4, 'score')
            sheet.write(cnt-1, 5, 888)
            print cnt
            cnt += 1
            continue
          for i, feat in enumerate(feats):
            # if len(feats) > 1:
            #   score = 777
            # else:
            diff = np.subtract(f0, feat)
            score = np.sum(np.square(diff))

            sheet.write(cnt-1, 0, 'cnt')
            sheet.write(cnt-1, 1, cnt)
            sheet.write(cnt-1, 2, 'name')
            sheet.write(cnt-1, 3, face)
            sheet.write(cnt-1, 4, 'score')
            sheet.write(cnt-1, 5, score.item())
            print cnt
            cnt += 1
            break
    # if cnt > 50:
    #   break
  excel.save(r'D:/face_recognition/MXNet/LFW_FRR_excel.xls')



def YTF():
  args = get_parser()
  face_identify = FaceIdendify(args)
  fileRes = r'YTF_result.txt'
  #excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
  #sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
  DBPath = 'D:/face_recognition/YTF/'
  cntTotal = 0
  cntNoface = 0
  cntSmall109 = 0
  cntLarge109 = 0
  cntSmall12 = 0
  cntLarge12 = 0
  cntName = 0
  #history_path = 'D:/testImage/history/'
  names = os.listdir(DBPath)
  for name in names:
    print name
    with open(fileRes,'a+') as fr:
      fr.write(name+'\r')
    namePath = DBPath+name
    cntName += 1
    videos = os.listdir(namePath)
    # flag = 0
    for video in videos:
      videoPath = namePath+'/'+video
      faces = os.listdir(videoPath)
      flag = 0
      for face in faces:
        facePath = videoPath+'/'+face
        #print facePath
        img = cv2.imread(facePath)
        feats, dets, points_all = face_identify.get_features(img)
        cntTotal +=1
        if flag == 0:
          if (not feats):
            cntNoface +=1
            continue
          else:
            for i, feat in enumerate(feats):
              f0 = feat
              flag = 1
              break
        else:
          if (not feats):
            cntNoface +=1
            continue
          else:
            for i, feat in enumerate(feats):
              diff = np.subtract(f0, feat)
              score = np.sum(np.square(diff))
              if score < 1.09:
                cntSmall109 += 1
              else:
                cntLarge109 += 1
              #break
              if score < 1.2:
                cntSmall12 += 1
              else:
                cntLarge12 += 1
              break
    per109 = (cntSmall109+cntNoface)*100.0/cntTotal
    per12 = (cntSmall12+cntNoface)*100.0/cntTotal
    print ("cntTotal={}, cntNoface={}, cntSmall109={}, cntLarge109={}, cntSmall12={}, cntLarge12={}, cntName={}, per109={}, per12={}".format(cntTotal,cntNoface,cntSmall109,cntLarge109,cntSmall12,cntLarge12,cntName,per109,per12))
    with open(fileRes,'a+') as fr:
      fr.write("cntTotal="+str(cntTotal)+" cntNoface="+str(cntNoface)+" cntSmall109="+str(cntSmall109)+" cntLarge109="+str(cntLarge109)+" cntSmall12="+str(cntSmall12)+" cntLarge12="+str(cntLarge12)+" cntName="+str(cntName)+" per109="+str(per109)+" per12="+str(per12)+'\r')

if __name__ == '__main__':
  YTF()
  #videoFile2videoFile()
  #args = get_parser()
  #face_identify = FaceIdendify(args)
  #face_identify.just_test()
  print ('done!')







