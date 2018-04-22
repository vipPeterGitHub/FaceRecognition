import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import time
import types

import face_embedding
import argparse
import cv2
import numpy as np
import cPickle as pickle


def get_parser():
  parser = argparse.ArgumentParser(description='face model test')
  # general
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
  parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
  parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

  parser.add_argument('--mtcnn', default=False, type=bool, help='whether use mtcnn, if not use dlib')
  parser.add_argument('--data_path', default='/home/peterhou/FaceRecognition/Database', type=str, help='the path of image base')
  parser.add_argument('--db_save_path', default='database/db_20170425_lab_qf.pickle', type=str, help='the save path of data base -- that is the feature of all images')
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
        else:
		    continue
        print file.split('.')[1][0]
        print ppath
        im=cv2.imread(ppath)
        feat=self.model.get_feature(im)[0]
        self.database[file.split('.')[0]]=feat
        cnt=cnt+1

    f = open(self.args.db_save_path, 'wb')
    pickle.dump(self.database,f,-1)
    f.close()
    print cnt, 'images have been done'
    print cnt

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
        anskey='Stranger/'+anskey
    fileRes = r'/home/peterhou/vic/data/videos/facevideos_out/20170528.txt'
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


if __name__ == '__main__':
  args = get_parser()
  face_identify = FaceIdendify(args)

  '''
  import ConfigParser
  cf=ConfigParser.ConfigParser()
  cf.read('config.ini')
  inputtype=int(cf.get("VideoInput","CameraType"))
  if(inputtype==1):
    capture=cv2.VideoCapture(cf.get("VideoInput","CameraIP"))
  if(inputtype==2):
    usbnum=int(cf.get("VideoInput","CameraUSB"))
    capture=cv2.VideoCapture(usbnum)
    capture.set(cv.CV_CAP_PROP_FRAME_WIDTH,1920)
    capture.set(cv.CV_CAP_PROP_FRAME_HEIGHT,1080)
  if(inputtype==3):
    filename=cf.get('VideoInput','VideoName')
    testname='/home/peterhou/FaceRecognition/Testdata/'+filename
    capture=cv2.VideoCapture(testname)
  '''

  #testvideo_path = '/home/peterhou/FaceRecognition/Testdata1/'
  testvideo_path = '/home/peterhou/vic/data/videos/test/'
  files= os.listdir(testvideo_path)
  #anspath='/home/peterhou/vic/data/videos/myout/'
  anspath = '/home/peterhou/vic/data/videos/facevideos_out/'
  fourcc = cv2.VideoWriter_fourcc(*"MPEG")
#  fps = capture.get(cv2.CAP_PROP_FPS)

  for filename in files:
    if filename.split('.')[-1] not in ['avi', 'mp4']:
      continue
    try:
      capture=cv2.VideoCapture(testvideo_path+filename)
    except:
      continue
    out = cv2.VideoWriter(anspath+filename, fourcc, 4.0, (1024,768))
  
    while capture.read():
	ret,img=capture.read()
	#cv2.imshow('Face Verification',img)
	#cv2.waitKey(10)
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
                  d = dets[i]
                  drawrec(minkey,minscore,img,d,filename.split('.')[0])
		#disp_db(db_path+minkey+'.png')
		img=cv2.resize(img,(1024,768))
		cv2.imshow('Face Verification',img)
		cv2.waitKey(10)
		out.write(img)
    capture.release()
    out.release()
  cv2.destroyAllWindows()




