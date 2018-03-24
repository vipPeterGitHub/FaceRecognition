#coding=utf-8
from PyQt4.QtCore import *
from PyQt4.QtGui import * 
import Queue
import scipy.io as sio  
from scipy import misc
import matplotlib.pyplot as plt  
#import pylab as pl
import numpy as np  
import caffe
#import sys

#caffe_root = '/home/peterhou/caffe-master/'
import sys
#sys.path.append(caffe_root+'python')
#sys.path.append(caffe_root+'python/caffe')

#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

#sys.path.append('/usr/lib/python2.7/dist-packages')
import os
import dlib
import glob
#from skimage import io
import cPickle as pickle
#from sklearn import svm
from PIL import Image
import time
#import cv2
#import cv2 as cv
import math

from scipy.io import savemat
from joint_bayesian import *
from sklearn.preprocessing import normalize
#from sklearn.lda import LDA
import os
import get_feature
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA
from skimage import transform as tf
from skimage.transform import warp
import types

import ConfigParser
cf=ConfigParser.ConfigParser()
cf.read('config.ini')



###########gui##########
import Tkinter as tk
from Tkinter import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil 
from PIL import Image, ImageTk
import threading
import multiprocessing
from multiprocessing import Process, Pool
#########################


def feat_norm(feat):
    tmp=np.zeros(feat.shape)
    for i in range(tmp.shape[0]):
        tmp[i]=feat[i]**2
    repfeat=feat/np.sqrt(sum(tmp.transpose()))
    return repfeat

def getfusedfeat(shape,img):
    imgs=normSingle(shape,img)
    start=time.clock()
    INPUT=np.zeros((7,3,112,96));
    INPUT[0,0,:,:]=(imgs[0][:,:,0]-127.5)/128
    INPUT[0,1,:,:]=(imgs[0][:,:,1]-127.5)/128
    INPUT[0,2,:,:]=(imgs[0][:,:,2]-127.5)/128
	
    INPUT[1,0,:,:]=(imgs[1][:,:,0]-127.5)/128
    INPUT[1,1,:,:]=(imgs[1][:,:,1]-127.5)/128
    INPUT[1,2,:,:]=(imgs[1][:,:,2]-127.5)/128
	
    INPUT[2,0,:,:]=(imgs[2][:,:,0]-127.5)/128
    INPUT[2,1,:,:]=(imgs[2][:,:,1]-127.5)/128
    INPUT[2,2,:,:]=(imgs[2][:,:,2]-127.5)/128\
	
    INPUT[3,0,:,:]=(imgs[3][:,:,0]-127.5)/128
    INPUT[3,1,:,:]=(imgs[3][:,:,1]-127.5)/128
    INPUT[3,2,:,:]=(imgs[3][:,:,2]-127.5)/128
	
    INPUT[4,0,:,:]=(imgs[4][:,:,0]-127.5)/128
    INPUT[4,1,:,:]=(imgs[4][:,:,1]-127.5)/128
    INPUT[4,2,:,:]=(imgs[4][:,:,2]-127.5)/128
	
    INPUT[5,0,:,:]=(imgs[5][:,:,0]-127.5)/128
    INPUT[5,1,:,:]=(imgs[5][:,:,1]-127.5)/128
    INPUT[5,2,:,:]=(imgs[5][:,:,2]-127.5)/128
	
    INPUT[6,0,:,:]=(imgs[6][:,:,0]-127.5)/128
    INPUT[6,1,:,:]=(imgs[6][:,:,1]-127.5)/128
    INPUT[6,2,:,:]=(imgs[6][:,:,2]-127.5)/128
	
    net_fused.blobs['data'].data[...]=INPUT
    out = net_fused.forward()
    feat_wholeface=feat_norm(net_fused.blobs['wholeface/fc5'].data[0].reshape(1,256))
    feat_ctf=feat_norm(net_fused.blobs['ctf/fc5'].data[0].reshape(1,256))
    feat_le=feat_norm(net_fused.blobs['le/fc5'].data[0].reshape(1,256))
    feat_re=feat_norm(net_fused.blobs['re/fc5'].data[0].reshape(1,256))
    feat_eye=feat_norm(net_fused.blobs['eye/fc5'].data[0].reshape(1,256))
    feat_mouth=feat_norm(net_fused.blobs['mouth/fc5'].data[0].reshape(1,256))
    feat_downmouth=feat_norm(net_fused.blobs['downmouth/fc5'].data[0].reshape(1,256))
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    print "get_featur time: ",time.clock()-start
    return feat

def getfeat_github(net,img):
    INPUT=np.zeros((3,112,96));
    INPUT[0,:,:]=(img[:,:,0]-127.5)/128
    INPUT[1,:,:]=(img[:,:,1]-127.5)/128
    INPUT[2,:,:]=(img[:,:,2]-127.5)/128
    net.blobs['data'].data[0]=INPUT
    out = net.forward()
    feat=net.blobs['fc5'].data[0].reshape(1,256)
    tmp=np.zeros(feat.shape)
    for i in range(tmp.shape[0]):
        tmp[i]=feat[i]**2
    repfeat=feat/np.sqrt(sum(tmp.transpose()))
    return repfeat

def getallfeat(shape,img):
    start=time.clock()
    imgs=normSingle(shape,img)
    end=time.clock()
    print ("normSingle time is {}".format(end-start))
    feat_wholeface=getfeat_github(net_wholeface,imgs[0])
    feat_ctf=getfeat_github(net_ctf,imgs[1])
    feat_le=getfeat_github(net_le,imgs[2])
    feat_re=getfeat_github(net_re,imgs[3])
    feat_eye=getfeat_github(net_eye,imgs[4])
    feat_mouth=getfeat_github(net_mouth,imgs[5])
    feat_downmouth=getfeat_github(net_downmouth,imgs[6])
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    return feat

def getallfeatNew(shape,img,net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth):
    start=time.clock()
    imgs=normSingle(shape,img)
    end=time.clock()
    print ("normSingle time is {}".format(end-start))
    feat_wholeface=getfeat_github(net_wholeface,imgs[0])
    feat_ctf=getfeat_github(net_ctf,imgs[1])
    feat_le=getfeat_github(net_le,imgs[2])
    feat_re=getfeat_github(net_re,imgs[3])
    feat_eye=getfeat_github(net_eye,imgs[4])
    feat_mouth=getfeat_github(net_mouth,imgs[5])
    feat_downmouth=getfeat_github(net_downmouth,imgs[6])
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    return feat


#fileRes = r'D:/face_recognition/Simple_Src/norm.txt'

def normSingle(shape,img):
    #with open(fileRes,'a+') as fr:
    #	fr.write("No. {} img is {}".format(num,img)+'\r')
    sevens=np.zeros((9,2))
    keyp=np.zeros((8,2))
    index=[30,39,36,42,45,48,54,37,43] # nt,lei,leo,rei,reo,lm,rm
    for i in range(len(index)):
        cord=str(shape.part(index[i])).split(',')
        x=int(cord[0][1:])
        y=int(cord[1][:-1])
        sevens[i][0]=x
        sevens[i][1]=y
        #print ("No. {} x = {}, y = {}".format(num,x,y))
    #print sevens[2]
    #print img[145,102]
    #img[int(sevens[7][1]),int(sevens[7][0])]=[255,255,255]
    #cv2.imwrite('dafd.jpg',img)
    wholeface_index= np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],[33.5493,92.3655],[62.7299,92.2041]])
    oriimg=np.array([(sevens[1]+sevens[2])/2,(sevens[3]+sevens[4])/2,sevens[0],sevens[5],sevens[6]])
    mapping=tf.estimate_transform('similarity', wholeface_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    wholeface=alignimg.astype(np.float64)*256

    ctf_index= np.array([[17,30],[88,30],[46,80],[20,100],[80,100]])
    oriimg=np.array([(sevens[1]+sevens[2])/2,(sevens[3]+sevens[4])/2,sevens[0],sevens[5],sevens[6]])
    mapping=tf.estimate_transform('similarity', ctf_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    ctf=alignimg.astype(np.float64)*256

    eye_index= np.array([[16,66],[80,66],[48,100]])
    oriimg=np.array([(sevens[1]+sevens[2])/2,(sevens[3]+sevens[4])/2,sevens[0]])
    mapping=tf.estimate_transform('similarity', eye_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    eye=alignimg.astype(np.float64)*256

    downmouth_index= np.array([[16,66],[80,66],[48,35]])
    oriimg=np.array([sevens[5],sevens[6],sevens[0]])
    mapping=tf.estimate_transform('similarity', downmouth_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    downmouth=alignimg.astype(np.float64)*256

    mouth_index= np.array([[16,37],[80,37],[48,6]])
    oriimg=np.array([sevens[5],sevens[6],sevens[0]])
    mapping=tf.estimate_transform('similarity', mouth_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    mouth=alignimg.astype(np.float64)*256

    re_index= np.array([[10,66],[42,66],[20,59]])
    oriimg=np.array([sevens[3],sevens[4],sevens[8]])
    mapping=tf.estimate_transform('similarity', re_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    re=alignimg.astype(np.float64)*256

    le_index= np.array([[54,66],[86,66],[64,59]])
    oriimg=np.array([sevens[2],sevens[1],sevens[7]])
    mapping=tf.estimate_transform('similarity', le_index, oriimg)
    alignimg=warp(img, mapping,output_shape=(112, 96))
    le=alignimg.astype(np.float64)*256
    
    #with open(fileRes,'a+') as fr:
    #	fr.write("No. {} wholeface is {}".format(num,wholeface)+'\r'+"No. {} ctf is {}".format(num,ctf)+'\r'+"No. {} le is {}".format(num,le)+'\r'+"No. {} re is {}".format(num,re)+'\r'+"No. {} eye is {}".format(num,eye)+'\r'+"No. {} mouth is {}".format(num,mouth)+'\r'+"No. {} downmouth is {}".format(num,downmouth)+'\r')

    return [wholeface,ctf,le,re,eye,mouth,downmouth]

def getthresh(feat1,feat2):
    Lx=np.sqrt(feat1.dot(feat1.transpose()))
    Ly=np.sqrt(feat2.dot(feat2.transpose()))
    cos_angle=feat1.dot(feat2.transpose())/(Lx*Ly)
    angle=np.arccos(cos_angle)
    return angle

#L2 distance
def getthreshL2(feat1,feat2):
    tmp=feat1.transpose()-feat2.transpose()
    for i in range(tmp.shape[0]):
        tmp[i]=tmp[i]**2
    return np.sqrt(sum(tmp))

###################################################################
#fileRes = r'D:/face_recognition/Testdata/20170528.txt'

##JB##
def getscore(A,G,feat,database):
    #global num
    #global anskey
    maxscore=-1000.
    anskey='nobody'
    for key,value in database.items():
        score=get_ratios(A,G,feat,value)
        if(score>maxscore):
            maxscore=score
            anskey=key
    if(maxscore<245):#id card: 140, multiface: 245
        anskey='Stranger'
    #with open(fileRes,'a+') as fr:
    #	fr.write('No. '+ str(num) +' anskey = '+str(anskey)+'  maxscore = '+str(maxscore)+'\r')
    #print ('No. '+ str(num) +' anskey = '+str(anskey)+'  maxscore = '+str(maxscore))
    #num = num + 1
    return anskey,maxscore
#######
def get_ratios(A, G, F1, F2):
    ratio = Verify(A, G, F1, F2)
    return ratio

def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(np.dot(np.transpose(x1), G), x2)
    return float(ratio)

def drawrec(minkey,minscore,img,d,video_name):
    if(minkey in colorbook.keys()):
        color=colorbook[minkey]
    else:
        color=(255,255,255)
    cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),color,2)   
    fontsize=(d.right()-d.left())/100.
    sidebar=int(round((d.right()-d.left())/15.))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,minkey,(d.left()+sidebar,d.top()-sidebar*20),font,0.8,color,1,False)
    cv2.putText(img,'%2f'%(minscore),(d.left()+sidebar,d.top()-sidebar*10),font,0.8,color,1,False)
    cv2.putText(img,video_name,(d.left()+sidebar,d.top()-sidebar*15),font,0.8,(255,0,0),1,False)
    
    #font=cv2.initFont(cv2.CV_FONT_HERSHEY_COMPLEX,0.1,fontsize,0,1,cv2.CV_AA)
    #cv2.putText(cv2.fromarray(img),minkey,(d.left()+sidebar,d.top()+sidebar*2),font,color)
    #cv.PutText(cv.fromarray(img),'%2f'%(minscore),(d.left()+sidebar,d.top()+sidebar*4),font,color)
    #cv.PutText(cv.fromarray(img),video_name,(d.left()+sidebar,d.top()+sidebar*6),font,(255,0,0))
    



predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Import caffe model
#caffe_root = cf.get("Parameters","caffe_root")
#sys.path.insert(0, caffe_root + 'python')
#import caffe


caffe.set_mode_gpu()
#caffe.set_device(0)

'''
global net_ctf,net_downmouth,net_eye,net_le,net_re,net_wholeface,net_mouth
net_wholeface = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/wholeface_iter_28000.caffemodel',
                    caffe.TEST)
net_ctf = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/ctf_iter_28000.caffemodel',
                    caffe.TEST)
net_le = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/le_iter_28000.caffemodel',
                    caffe.TEST)
net_re = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/re_iter_28000.caffemodel',
                    caffe.TEST)
net_eye = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/eye_iter_28000.caffemodel',
                    caffe.TEST)
net_mouth = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/mouth_iter_28000.caffemodel',
                    caffe.TEST)
net_downmouth = caffe.Net('../Models/face_deploy.prototxt',
                    '../Models/downmouth_iter_28000.caffemodel',
                    caffe.TEST)
'''


# Import Lab database
#f = open('../Models/db_20170425_lab_qf.pickle','rb')
#db_qf=pickle.load(f)
#f.close()

# Define color book
colorbook={'Stranger':(0,0,0)}

dispTime=100000
start_time=time.time()

count=0
'''
global capture
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
    testname='../Testdata/'+filename
    capture=cv2.VideoCapture(testname)
'''
'''
#####result######
anspath='../videos_images/results/'
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
fourcc = cv2.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter(anspath+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi', fourcc, 12, (1024,768))
out = cv2.VideoWriter(anspath+filename, fourcc, 8, (1024,768))
###############
'''

def get_A_G_testset_feature(result_fold='../Jb/'):
    """
    get matrix A,G and feature of lfw dataset
    :return: A,G is para in plda,data[i] is ith image's feature in lfw dataset
    """
    with open(result_fold + 'A.pkl', 'rb') as f:
        A = pickle.load(f)
    with open(result_fold + 'G.pkl', 'rb') as f:
        G = pickle.load(f)
    return A, G

A,G=get_A_G_testset_feature()


#clt_pca = joblib.load('../Jb/'+"pca_model.m")

saveMeanPath = '../Jb/mean.mat'
saveCompoTPath = '../Jb/compoT.mat'
#savemat(saveMeanPath,{'mean':mean})
#savemat(saveCompoTPath,{'compoT':compoT})

mean = loadmat(saveMeanPath)['mean']
compoT = loadmat(saveCompoTPath)['compoT']

#f = open('../Models/db_20170425_lab_qf.pickle','rb')
f = open('../Models/20180309_database_hou.pickle','rb')
#f = open('../Models/20180309_database_hou_simple.pickle','rb')
#f = open('../Models/610324199510253457_DB.pickle','rb')
db_qf=pickle.load(f)
f.close()


filename = '20170528.avi'
#capture = cv2.VideoCapture('../Testdata/20170528.avi')
#capture = cv2.VideoCapture(0)




def faceRec(net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth, img, gray, d, DB = db_qf):
    begin = time.clock()
    shape = predictor(gray, d)
    feat_pca = np.dot(getallfeatNew(shape,img,net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth)-mean,compoT)
    end1 = time.clock()
    print ("shape to featpca time is {}".format(end1-begin))
    minkey,minscore=getscore(A,G,feat_pca,DB)
    print ("minscore is {}, minkey is {}".format(str(int(minscore)),minkey))
    end = time.clock()
    print ("Face calculation time is {}".format(end-begin))
    return minkey,minscore

'''
def faceRec(img, gray, d, DB = db_qf):
    begin = time.clock()
    shape = predictor(gray, d)
    feat_pca = np.dot(getallfeat(shape,img,)-mean,compoT)
    end1 = time.clock()
    print ("shape to featpca time is {}".format(end1-begin))
    minkey,minscore=getscore(A,G,feat_pca,DB)
    print ("minscore is {}, minkey is {}".format(str(int(minscore)),minkey))
    end = time.clock()
    print ("Face calculation time is {}".format(end-begin))
    return minkey,minscore
'''


##############################################################################
'''
def convImg(img,w,h):
    resize_im=cv2.resize(img,(w,h))
    cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
    return byte_im

class MainWindow(QWidget):
    def __init__(self, qFace, qDet, qGray, qImg):
        super(MainWindow, self).__init__()
        self.status = 0
        self.playButton = QPushButton("Play")
        self.cancelButton = QPushButton("Cancel")
        self.exitbtn = QPushButton('exit')
        self.image = QImage()
        self.imageLabel = QLabel('')
        self.faceLabel = QLabel('')
        self.idLabel = QLabel('')
        self.textLabel = QLabel("Hello World!")
        self.textLabel.setFixedWidth(1170)  
        self.textLabel.setFixedHeight(100)  
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setFont(QFont("Roman times",50,QFont.Bold))

        self.twoFace = QVBoxLayout()
        self.twoFace.addWidget(self.faceLabel)
        #self.twoFace.addStretch(1)
        self.twoFace.addWidget(self.idLabel)

        self.videoImage = QHBoxLayout()
        self.videoImage.addWidget(self.imageLabel)
        #self.videoImage.addStretch()
        self.videoImage.addLayout(self.twoFace)

        self.buttons = QHBoxLayout()
        #hbox.addStretch(1)
        self.buttons.addWidget(self.playButton)
        self.buttons.addWidget(self.cancelButton)
        self.buttons.addWidget(self.exitbtn)

        self.whole = QVBoxLayout()
        self.whole.addLayout(self.videoImage)
        self.whole.addWidget(self.textLabel)
        self.whole.addLayout(self.buttons)
        
        self.setLayout(self.whole)

        self.showImageLabel('nopic.jpg')
        self.showFaceLabel('nopic.jpg')
        self.showIdLabel('nopic.jpg')
        self.qFace = qFace
        self.qDet = qDet
        self.qGray = qGray
        self.qImg = qImg
        self.playtimer = Timer("videoPlay", self.qFace)
        self.facetimer = FaceTimer(self.qFace, "facePlay", self.qDet, self.qGray, self.qImg)
        self.idcardtimer = IdcardTimer("idcardPlay")
        self.texttimer = TextTimer(self.qDet, self.qGray, self.qImg, "textPlay")

        self.connect(self.texttimer, SIGNAL("textPlay"), self.playText)

        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)
        self.connect(self.facetimer, SIGNAL("facePlay"), self.playFace)
        self.connect(self.idcardtimer, SIGNAL("idcardPlay"), self.playIdcard)
        self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        self.connect(self.playButton, SIGNAL("clicked()"), self.VideoPlayPause)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self.exit)
        self.setWindowTitle('Face Recognition')
        self.resize(1170, 800)

    def playText(self, s):
        print s
        self.textLabel.setText(s)

    def playVideo(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
    def playFace(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.faceLabel.setPixmap(QPixmap.fromImage(self.image))
    def playIdcard(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.idLabel.setPixmap(QPixmap.fromImage(self.image))
        
    def VideoPlayPause(self):
        #self.status, playstr, capturestr = ((1, 'pause', 'capture'), (0, 'play', 'capture'), (1, 'pause', 'capture'))[self.status]#三种状态分别对应的显示、处理
        self.status, playstr = ((1, 'pause'), (0, 'play'))[self.status]
        self.playButton.setText(playstr)
        if self.status is 1:
            #self.timer.stop()
            self.playtimer.start()
            self.facetimer.start()
            self.idcardtimer.start()
            self.texttimer.start()
        else:
            self.playtimer.stop()
            self.facetimer.stop()
            self.idcardtimer.stop()
            self.texttimer.stop()

    def showImageLabel(self,im):
        orig_im = cv2.imread(im)
        #resize_im=cv2.resize(orig_im,(900,630))
        #cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
        #pil_im = Image.fromarray(cv2_im)
        #byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
        byte_im = convImg(orig_im,900,630)
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        #self.imageLabel.setScaledContents(True)
    def showFaceLabel(self,im):
        orig_im = cv2.imread(im)
        byte_im = convImg(orig_im,220,300)
        self.image.loadFromData(QByteArray(byte_im))
        self.faceLabel.setPixmap(QPixmap.fromImage(self.image))
        #self.faceLabel.setScaledContents(True)
    def showIdLabel(self,im):
        orig_im = cv2.imread(im)
        byte_im = convImg(orig_im,220,300)
        self.image.loadFromData(QByteArray(byte_im))
        self.idLabel.setPixmap(QPixmap.fromImage(self.image))
        #self.idLabel.setScaledContents(True)

class TextTimer(QThread):
    
    def __init__(self, qDet = Queue.Queue(4), 
                        qGray = Queue.Queue(4), qImg = Queue.Queue(4),
                        signal = "updateTime", parent=None):
        super(TextTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False

        while True:
            if self.stoped:
                return
            if not qDet.empty():
                d= qDet.get(True,3)
                gray = qGray.get(True,3)
                img = qImg.get(True,3) #get*3<0.001s
                begin = time.clock()
                shape = predictor(gray, d) #0.06s
                begin = time.clock()
                feat_pca = np.dot(getallfeat(shape,img)-mean,compoT) #4s
                end = time.clock()
                print ("processing time is {}".format(end-begin))
                minkey,minscore=getscore(A,G,feat_pca,db_qf)
                print ("minkey is {}".format(minkey))
                s = str(minscore)+" "+minkey
                self.emit(SIGNAL(self.signal),s)
                #time.sleep(0.04)

    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class Timer(QThread):
    
    def __init__(self, signal = "updateTime", qFace = Queue.Queue(4), parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        #camera = cv2.VideoCapture('../Testdata/20170614.avi')
        #camera = cv2.VideoCapture(1)
        
        while True:
            if self.stoped:
                return
            ret, face = capture.read()
            if not qFace.full():
                qFace.put(face)
            byte_im = convImg(face,900,630)
            self.emit(SIGNAL(self.signal),byte_im)
            time.sleep(0.04)
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class FaceTimer(QThread):
    
    def __init__(self, qFace = Queue.Queue(4), signal = "updateTime", 
                        qDet = Queue.Queue(4), qGray = Queue.Queue(4), 
                        qImg = Queue.Queue(4), parent=None):
        super(FaceTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()


    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            if self.stoped:
                return
            if not qFace.empty():
                img = qFace.get(True,3)
                #print ("type is {}".format(type(img)))
                if isinstance(img, types.NoneType):
                    img = cv2.imread("nopic.jpg")
                else:
                    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    print gray.shape
                    dets = detector(gray)
                    if(len(dets)==0):
                        print "No face detected"
                        minkey = 'Stranger'
                        img = cv2.imread("nopic.jpg")
                    else:
                        for k, d in enumerate(dets):
                            if not qDet.full():
                                qDet.put(d)
                                qGray.put(gray)
                                qImg.put(img)
                            [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                            if x1<0:
                                x1 = 10
                            #print ("img_shape = {}".format(img.shape))
                            img = img[y1:y2,x1:x2,:]
                            #print ("x1={}x2={}y1{}y2{}".format(x1,x2,y1,y2))
                            break

                byte_im = convImg(img,220,300)
                self.emit(SIGNAL(self.signal),byte_im)
                time.sleep(0.04)
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class IdcardTimer(QThread):
    
    def __init__(self, signal = "updateTime", parent=None):
        super(IdcardTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            if self.stoped:
                return
            face = cv2.imread('hou.bmp')
            byte_im = convImg(face,220,300)
            self.emit(SIGNAL(self.signal),byte_im)
            time.sleep(0.04)
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

qFace = Queue.Queue(10)
qDet = Queue.Queue(10)
qGray = Queue.Queue(10)
qImg = Queue.Queue(10)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow(qFace, qDet, qGray, qImg)
    mainwindow.show()
    sys.exit(app.exec_())
'''
###########################################################################


class FaceDetector(object):
    '''
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a caffe version
    '''
    def __init__(self,
                 minsize = 20,
                 threshold = [0.6, 0.7, 0.7],
                 factor = 0.709,
                 fastresize = False,
                 gpuid = 0):
        
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor
        self.fastresize = fastresize
        
        model_P = './model/det1.prototxt'
        weights_P = './model/det1.caffemodel'
        model_R = './model/det2.prototxt'
        weights_R = './model/det2.caffemodel'
        model_O = './model/det3.prototxt'
        weights_O = './model/det3.caffemodel'
        
        caffe.set_mode_gpu()
        #caffe.set_device(gpuid)
        
        self.PNet = caffe.Net(model_P, weights_P, caffe.TEST) 
        self.RNet = caffe.Net(model_R, weights_R, caffe.TEST)
        self.ONet = caffe.Net(model_O, weights_O, caffe.TEST)     
        

    def bbreg(self,boundingbox,reg):
    
        '''Calibrate bounding boxes'''
        
        if reg.shape[1]==1:
            reg = np.shape(reg,(reg.shape[2],reg.shape[3])).T
        w = boundingbox[:,2]-boundingbox[:,0]+1
        h = boundingbox[:,3]-boundingbox[:,1]+1
        boundingbox[:,0:4] = np.reshape(np.hstack((boundingbox[:,0]+reg[:,0]*w, boundingbox[:,1]+reg[:,1]*h, boundingbox[:,2]+reg[:,2]*w, boundingbox[:,3]+reg[:,3]*h)),(4,w.shape[0])).T
    
        return boundingbox
    
    def nms(self,dets, thresh,type='Union'):
        
        if dets.shape[0]==0:
            keep = []
            return keep

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
    
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
    
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type=='Min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])  
            else:
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
    
        return keep
        
    def rerec(self,bboxA):
        
        '''Convert bboxA to square'''
        
        h = bboxA[:,3]-bboxA[:,1]
        w = bboxA[:,2]-bboxA[:,0]
        l = np.concatenate((w,h)).reshape((2,h.shape[0]))
        l = np.amax(l, axis=0) 
        bboxA[:,0] = bboxA[:,0] + w*0.5 -l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 -l*0.5
        bboxA[:,2] = bboxA[:,0] + l
        bboxA[:,3] = bboxA[:,1] + l
    
        return bboxA
    
    def sort_rows_by_icol1(self,inarray):

        idex=np.lexsort([inarray[:,0],inarray[:,1]])
        a_sort=inarray[idex,:]
        return a_sort
    
    
    def generateBoundingBox(self,map,reg,scale,threshold):
    
        '''Use heatmap to generate bounding boxes'''
        
        stride=2;
        cellsize=12;
        boundingbox=[];
        
        map = map.T
        dx1=reg[:,:,0].T
        dy1=reg[:,:,1].T
        dx2=reg[:,:,2].T
        dy2=reg[:,:,3].T
  
        [y,x]=np.where(map>=threshold)
        y = np.reshape(y,(len(y),1))
        x = np.reshape(x,(len(y),1))
        a = np.where(map.flatten(1)>=threshold)

        if y.shape[0]==1:
            y=y.T
            x=x.T
            score=np.reshape(map.flatten(1)[a[0]],(1,1))
            dx1=dx1.T
            dy1=dy1.T
            dx2=dx2.T
            dy2=dy2.T
        else:

            score=map.flatten(1)[a[0]]
            score=np.reshape(score, (a[0].shape[0],1))
            
        dx1N=np.reshape(dx1.flatten(1)[a[0]], (a[0].shape[0],1))
        dy1N=np.reshape(dy1.flatten(1)[a[0]], (a[0].shape[0],1))
        dx2N=np.reshape(dx2.flatten(1)[a[0]], (a[0].shape[0],1))
        dy2N=np.reshape(dy2.flatten(1)[a[0]], (a[0].shape[0],1))  
        
        reg=np.hstack((dx1N,dy1N,dx2N,dy2N))
        
        if  reg.shape[0]==0:
            reg = np.zeros(shape=(0,3))
        
        boundingbox=np.hstack((y,x))
        boundingbox = self.sort_rows_by_icol1(boundingbox)
        boundingbox=np.hstack((((stride*boundingbox+1)/scale-1).astype(int),(((stride*boundingbox+cellsize-1+1)/scale-1)).astype(int),score,reg))

        return boundingbox
    
    def pad(self,total_boxes,w,h):
    
        '''Compute the padding coordinates (pad the bounding boxes to square)'''
        
        tmpw=total_boxes[:,2]-total_boxes[:,0]+1
        tmph=total_boxes[:,3]-total_boxes[:,1]+1
        numbox=total_boxes.shape[0]
        
        dx = np.ones((numbox,))
        dy = np.ones((numbox,))
        
        edx = tmpw    
        edy = tmph
            
        x = total_boxes[:,0]
        y = total_boxes[:,1]
        ex = total_boxes[:,2]
        ey = total_boxes[:,3]
        
        tmp = np.where(ex>w)
        edx[tmp] = -ex[tmp] + w + tmpw[tmp]
        ex[tmp] = w
        
        tmp = np.where(ey>h)
        edy[tmp]= -ey[tmp] + h + tmph[tmp]
        ey[tmp] = h
        
        tmp = np.where(x < 1)
        dx[tmp] = 2-x[tmp]
        x[tmp] = 1  
        
        tmp = np.where(y < 1)
        dy[tmp] = 2-y[tmp]
        y[tmp] = 1
        
        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph
    
        
    def LoadNet(self,model,weights):
        caffe.set_mode_gpu()
        #caffe.set_device(0)
        Net = caffe.Net(model, weights, caffe.TEST)
        return Net
    
    def detectface(self,img):

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.fastresize:
            im_data=(np.float32(img)-127.5)*0.0078125
        
        
        factor_count=0
        total_boxes=[]
        points=[]
        h=img.shape[0]
        w=img.shape[1]

        minl=min(w,h)
        m=12.0/self.minsize
        minl=minl*m
        # creat scale pyramid
        scales=[]
        while (minl>=12.0):
            scales.append(m*(math.pow(self.factor,factor_count)))
            minl=minl*self.factor
            factor_count=factor_count+1

        total_boxes = np.zeros(shape=(0,9))

        for scale in scales:
        
            hs=int(math.ceil(h*scale))
            ws=int(math.ceil(w*scale))
            if self.fastresize:
                im_data_out = cv2.resize(im_data,(ws, hs),interpolation=cv2.INTER_NEAREST)

            else:
                im_data_out = (cv2.resize(img,(ws, hs),interpolation=cv2.INTER_NEAREST) - 127.5)*0.0078125
            im_data_out = im_data_out[None,:] 
            im_data_out = im_data_out.transpose((0,3,2,1)) 
            self.PNet.blobs['data'].reshape(1,3,ws,hs)
            out = self.PNet.forward_all( data = im_data_out )
            
            
            map = out['prob1'][0].transpose((2,1,0))[:,:,1]
            reg = out['conv4-2'][0].transpose((2,1,0))
            boxes = self.generateBoundingBox(map,reg,scale,self.threshold[0])
            
            pick = self.nms(boxes, 0.5)
            boxes = boxes[pick,:]
            if boxes.shape[0]!=0:
                total_boxes = np.concatenate((total_boxes,boxes),axis=0)
        

        if total_boxes is not None:
            pick = self.nms(total_boxes, 0.7)
            total_boxes = total_boxes[pick,:]
            regw=total_boxes[:,2]-total_boxes[:,0];
            regh=total_boxes[:,3]-total_boxes[:,1];
            total_boxes = np.concatenate((total_boxes[:,0]+total_boxes[:,5]*regw, total_boxes[:,1]+total_boxes[:,6]*regh, total_boxes[:,2]+total_boxes[:,7]*regw, total_boxes[:,3]+total_boxes[:,8]*regh, total_boxes[:,4])).reshape((5,regw.shape[0]))   
            total_boxes = total_boxes.T
            total_boxes=self.rerec(total_boxes)
            total_boxes[:,0:4]=total_boxes[:,0:4].astype(int)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes,w,h)
            
        numbox = total_boxes.shape[0]

        
        if  numbox > 0:    
            #second stage
            tempimg =  np.zeros((24,24,3,numbox))
            for k in range(numbox):
                tmp =  np.zeros((int(tmph[k]),int(tmpw[k]),3))
                tmp[int(dy[k]-1):int(edy[k]),int(dx[k]-1):int(edx[k]),:]=img[int(y[k]-1):int(ey[k]),int(x[k]-1):int(ex[k]),:] 
                tempimg[:,:,:,k]= cv2.resize(tmp,(24, 24),interpolation=cv2.INTER_NEAREST)
            tempimg = (tempimg-127.5)*0.0078125
            tempimg = tempimg.transpose((3,2,1,0)) 
            self.RNet.blobs['data'].reshape(numbox,3,24,24)
            out = self.RNet.forward_all( data = tempimg )        

            score=out['prob1'][:,1]   ###why need to squeeze?
            pas = np.where(score>self.threshold[1])            
            total_boxes = np.hstack((total_boxes[pas[0],0:4], np.reshape(score[pas[0]],(len(pas[0]),1))))
            mv = out['conv5-2'][pas[0],:]

            if total_boxes is not None:
                pick = self.nms(total_boxes, 0.7)
                total_boxes = total_boxes[pick,:]  
                total_boxes=self.bbreg(total_boxes, mv[pick,:])
                total_boxes=self.rerec(total_boxes)
                
            numbox = total_boxes.shape[0]
        
            if  numbox > 0: 
                # third stage
                total_boxes = total_boxes.astype(int)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes,w,h)
                tempimg =  np.zeros((48,48,3,numbox))
                for k in range(numbox):
                    tmp =  np.zeros((tmph[k],tmpw[k],3))
                    tmp[int(dy[k]-1):int(edy[k]),int(dx[k]-1):int(edx[k]),:]=img[int(y[k]-1):int(ey[k]),int(x[k]-1):int(ex[k]),:]
                    tempimg[:,:,:,k]= cv2.resize(tmp,(48, 48),interpolation=cv2.INTER_NEAREST)    
                tempimg = (tempimg-127.5)*0.0078125 
                tempimg = tempimg.transpose((3,2,1,0)) 
                self.ONet.blobs['data'].reshape(numbox,3,48,48)
                out = self.ONet.forward_all( data = tempimg ) 
        
                score = out['prob1'][:,1]
                points = out['conv6-3']
                pas = np.where(score>self.threshold[2])
                points = points[pas[0],:].T
                total_boxes = np.hstack((total_boxes[pas[0],0:4], np.reshape(score[pas[0]],(len(pas[0]),1))))
                mv = out['conv6-2'][pas[0],:]
                w=total_boxes[:,2]-total_boxes[:,0]+1
                h=total_boxes[:,3]-total_boxes[:,1]+1
                points[0:5,:] = np.tile(np.reshape(w,(1,w.shape[0])),[5,1])*points[0:5,:]+np.tile(np.reshape(total_boxes[:,0],(1,total_boxes.shape[0])),[5,1])-1
                points[5:10,:] = np.tile(np.reshape(h,(1,h.shape[0])),[5,1])*points[5:10,:]+np.tile(np.reshape(total_boxes[:,1],(1,total_boxes.shape[0])),[5,1])-1
                if total_boxes is not None:
                    total_boxes=self.bbreg(total_boxes, mv[:,:])
                    pick = self.nms(total_boxes, 0.7, 'Min')
                    total_boxes = total_boxes[pick,:]
                    points = points[:,pick]
            numbox = total_boxes.shape[0]       
        return total_boxes,points,numbox
    


def databaseUpdate():

    caffe.set_mode_gpu()
    net_wholeface = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/wholeface_iter_28000.caffemodel',
                        caffe.TEST)
    net_ctf = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/ctf_iter_28000.caffemodel',
                        caffe.TEST)
    net_le = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/le_iter_28000.caffemodel',
                        caffe.TEST)
    net_re = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/re_iter_28000.caffemodel',
                        caffe.TEST)
    net_eye = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/eye_iter_28000.caffemodel',
                        caffe.TEST)
    net_mouth = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/mouth_iter_28000.caffemodel',
                        caffe.TEST)
    net_downmouth = caffe.Net('../Models/face_deploy.prototxt',
                        '../Models/downmouth_iter_28000.caffemodel',
                        caffe.TEST)

    print "Loading new done!"

    mtcnn_detector = FaceDetector(minsize = 80, gpuid = 0, fastresize = False)
    database = {'Name': 'Feature'};
    database.clear();
 
    path = "../Database" 
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
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        total_boxes,points,numbox = mtcnn_detector.detectface(im)
        dets=set()
        for i in range(numbox):
            dets.add(dlib.rectangle(int(total_boxes[i][0]),int(total_boxes[i][1]),int(total_boxes[i][2]),int(total_boxes[i][3]))) 
        if(len(dets)==0):
            print 'No face detected'

            continue
        for k, d in enumerate(dets):
            shape = predictor(gray, d) 
            feat=getallfeatNew(shape,im,net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth)
        #database[file.split('.')[0]]=clt_pca.transform(feat)
        database[file.split('.')[0]]=np.dot(feat-mean,compoT)
        cnt=cnt+1
        print cnt

    f = open('../Models/20180309_database_hou.pickle','wb')
    pickle.dump(database,f,-1)
    f.close()
    print cnt

def createDatabase(id_num,picPath,picklePath):# one picture database
    mtcnn_detector = FaceDetector(minsize = 80, gpuid = 0, fastresize = False)
    database = {'Name': 'Feature'};
    database.clear();

    im=cv2.imread(picPath)
    gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    total_boxes,points,numbox = mtcnn_detector.detectface(im)
    dets=set()
    for i in range(numbox):
        dets.add(dlib.rectangle(int(total_boxes[i][0]),int(total_boxes[i][1]),int(total_boxes[i][2]),int(total_boxes[i][3]))) 
    if(len(dets)==0):
        print 'No face detected and creating completed!'
    else:
        for k, d in enumerate(dets):
            shape = predictor(gray, d) 
            feat=getallfeat(shape,im)
        #database[file.split('.')[0]]=clt_pca.transform(feat)
        database[id_num]=np.dot(feat-mean,compoT)

        f = open(picklePath,'wb')
        pickle.dump(database,f,-1)
        f.close()
        print "DB file has been created! "

def compareFace(face,picklePath):
    f = open(picklePath,'rb')
    DB=pickle.load(f)
    f.close()
    gray=cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    for k, d in enumerate(dets):
        shape = predictor(gray, d) 
        #feat_pca=clt_pca.transform(getallfeat(shape,img))
        feat_pca = np.dot(getallfeat(shape,face)-mean,compoT)
        minkey,minscore=getscore(A,G,feat_pca,DB)

    print ("compare result is: Minkey = {}, Minscore = {}".format(minkey,minscore))
