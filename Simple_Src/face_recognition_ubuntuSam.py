import scipy.io as sio  
from scipy import misc
import matplotlib.pyplot as plt  
import pylab as pl
import numpy as np  
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import os
import dlib
import glob
from skimage import io
import cPickle as pickle
from sklearn import svm
from PIL import Image
import time
import cv2
import cv2.cv as cv
import math

from scipy.io import savemat
from joint_bayesian import *
from sklearn.preprocessing import normalize
from sklearn.lda import LDA
import os
import get_feature
from sklearn.externals import joblib

from skimage import transform as tf
from skimage.transform import warp

import ConfigParser
cf=ConfigParser.ConfigParser()
cf.read('config.ini')

    
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
    print end-start
    start=time.clock()
    feat_wholeface=getfeat_github(net_wholeface,imgs[0])
    end=time.clock()
    print end-start
    feat_ctf=getfeat_github(net_ctf,imgs[1])
    feat_le=getfeat_github(net_le,imgs[2])
    feat_re=getfeat_github(net_re,imgs[3])
    feat_eye=getfeat_github(net_eye,imgs[4])
    feat_mouth=getfeat_github(net_mouth,imgs[5])
    feat_downmouth=getfeat_github(net_downmouth,imgs[6])
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    return feat

def normSingle(shape,img):
    sevens=np.zeros((9,2))
    keyp=np.zeros((8,2))
    index=[30,39,36,42,45,48,54,37,43] # nt,lei,leo,rei,reo,lm,rm
    for i in range(len(index)):
        cord=str(shape.part(index[i])).split(',')
        x=int(cord[0][1:])
        y=int(cord[1][:-1])
        sevens[i][0]=x
        sevens[i][1]=y
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

##JB##
def getscore(A,G,feat,database):
    maxscore=-1000.
    anskey='nobody'
    for key,value in database.items():
        score=get_ratios(A,G,feat,value)
        if(score>maxscore):
            maxscore=score
            anskey=key
    if(maxscore<300):
        anskey='Stranger'
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
    fontsize=(d.right()-d.left())/200.
    sidebar=int(round((d.right()-d.left())/15.))
    font=cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX,0.1,fontsize,0,1,cv.CV_AA)
    cv.PutText(cv.fromarray(img),minkey,(d.left()+sidebar,d.top()+sidebar*2),font,color)
    cv.PutText(cv.fromarray(img),'%2f'%(minscore),(d.left()+sidebar,d.top()+sidebar*4),font,color)
    cv.PutText(cv.fromarray(img),video_name,(d.left()+sidebar,d.top()+sidebar*6),font,(255,0,0))
    



predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Import caffe model
caffe_root = cf.get("Parameters","caffe_root")
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()

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


# Import Lab database
f = open('../Models/db_20170425_lab_qf.pickle','rb')
db_qf=pickle.load(f)
f.close()

# Define color book
colorbook={'Stranger':(0,0,0)}

dispTime=100000
start_time=time.time()

count=0

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


#####result######
anspath='../videos_images/results/'
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter(anspath+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi', fourcc, 12, (1024,768))
out = cv2.VideoWriter(anspath+filename, fourcc, 8, (1024,768))
###############


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
clt_pca = joblib.load('../Jb/'+"pca_model.m")



while time.time()-start_time<dispTime and capture.read():
    ret,img=capture.read()
    #img = cv2.warpAffine(img,M,(cols,rows))
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Face Detector
    dets = detector(gray)
    if(len(dets)==0):
        print "No face detected"
        img=cv2.resize(img,(1024,768))
#        cv2.imshow('Face Verification',img)
#        cv2.waitKey(10)
        out.write(img)
        continue
    #else:
    	#out.write(cv2.resize(img,(1024,768)))
    # Landmark Detector & Get L2 Features
    #cv2.imwrite('window/%s.jpg'%count,img)
    for k, d in enumerate(dets):
        shape = predictor(gray, d) 
        feat_pca=clt_pca.transform(getallfeat(shape,img))
#         if(i==0):
#             feat_0=feat
#         else:
#             print getthresh(feat_0,feat)
#             feat_0=feat
        # Compare with Database
        minkey,minscore=getscore(A,G,feat_pca,db_qf)
        # Draw the Results      
        #if(minkey!='Stranger'):
        drawrec(minkey,minscore,img,d,filename.split('.')[0])
    #disp_db(db_path+minkey+'.png')
    img=cv2.resize(img,(1024,768))
    cv2.imshow('Face Verification',img)
    cv2.waitKey(10)
    #out.write(img)
capture.release()
cv2.destroyAllWindows()
out.release()
