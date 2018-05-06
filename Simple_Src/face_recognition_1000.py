import scipy.io as sio  
from scipy import misc
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import matplotlib.pyplot as plt  
import pylab as pl
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
from skimage import io
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
import xlwt
import shutil

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
    	#fr.write("No. {} wholeface is {}".format(num,wholeface)+'\r'+"No. {} ctf is {}".format(num,ctf)+'\r'+"No. {} le is {}".format(num,le)+'\r'+"No. {} re is {}".format(num,re)+'\r'+"No. {} eye is {}".format(num,eye)+'\r'+"No. {} mouth is {}".format(num,mouth)+'\r'+"No. {} downmouth is {}".format(num,downmouth)+'\r')

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
fileRes = r'D:/face_recognition/Testdata/2018_03_12_17_52_57_CZM.txt'

##JB##
def getscore(A,G,feat,database):
    global num
    maxscore=-1000.
    anskey='nobody'
    for key,value in database.items():
        score=get_ratios(A,G,feat,value)
        if(score>maxscore):
            maxscore=score
            anskey=key
    if(maxscore<300):
        #anskey='Stranger'
        anskey='Stranger_'+anskey
    with open(fileRes,'a+') as fr:
    	fr.write('No. '+ str(num) +' anskey = '+str(anskey)+'  maxscore = '+str(maxscore)+'\r')
    print ('No. '+ str(num) +' anskey = '+str(anskey)+'  maxscore = '+str(maxscore))
    num = num + 1
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
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,minkey,(d.left()+sidebar,d.top()-sidebar*5),font,1,color,1,False)
    cv2.putText(img,'%2f'%(minscore),(d.left()+sidebar,d.top()-sidebar*0),font,1,color,1,False)
    #cv2.putText(img,video_name,(d.left()+sidebar,d.top()-sidebar*10),font,1,(255,0,0),1,False)
    
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

#caffe.set_mode_gpu()
caffe.set_mode_gpu()
#caffe.set_device(0)

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
f = open('../Models/20180426_db600_20stars_beijing&GOT_caffe_hou.pickle','rb')
db_qf=pickle.load(f)
f.close()

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
print A.shape
print G.shape
print db_qf.keys()
print db_qf['GuanYue'].shape

#clt_pca = joblib.load('../Jb/'+"pca_model.m")

saveMeanPath = '../Jb/mean.mat'
saveCompoTPath = '../Jb/compoT.mat'
#savemat(saveMeanPath,{'mean':mean})
#savemat(saveCompoTPath,{'compoT':compoT})

mean = loadmat(saveMeanPath)['mean']
compoT = loadmat(saveCompoTPath)['compoT']


'''
fourcc = cv2.VideoWriter_fourcc(*"DIVX")#(*"XVID")
saveVideoPath = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'2018_03_19_17_52_53'+'.avi'
saveVideoPath2 = 'pinjie1.avi'
out = cv2.VideoWriter(saveVideoPath,fourcc,5,(640,480)) # 10 is speed. 640,480


filename = 'video1.avi'
capture = cv2.VideoCapture('D:/face_recognition/Testdata/0-200.avi')
#capture = cv2.VideoCapture(0)
'''
num = 1

#capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#apture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

def caffe1000():
    inputPath = "D:/face_recognition/Database1600/1000_input/"
    errorPicPath = "D:/face_recognition/Simple_Src/errorPairCaffe/"
    fileRes = r'D:/face_recognition/Simple_Src/1000_input_caffe.txt'
    excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
    cnt = 1
    faces = os.listdir(inputPath)
    for face in faces:
        facePath = inputPath+face
        img = cv2.imread(facePath)
        minkey,minscore = img2score(img)
        with open(fileRes,'a+') as fr:
            fr.write('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore)+'\r')

        print ('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore))
        sheet.write(cnt-1, 0, 'cnt')
        sheet.write(cnt-1, 1, cnt)
        sheet.write(cnt-1, 2, 'minkey')
        sheet.write(cnt-1, 3, minkey)
        sheet.write(cnt-1, 4, 'score')
        sheet.write(cnt-1, 5, minscore)
        if minscore>300:
            shutil.copyfile(facePath, errorPicPath+minkey+face)
            shutil.copyfile('D:/face_recognition/DatabaseFace/'+minkey+'.jpg', errorPicPath+minkey+'.jpg')
        print cnt
        cnt+=1
    excel.save(r'D:/face_recognition/Simple_Src/score1000_caffe.xls')


def caffeClassFaces():
    inputPath = "D:/face_recognition/testFile656/beijing_GOT_faces/"
    #errorPicPath = "D:/face_recognition/Database1600/errorPair/"
    labelPath = "D:/face_recognition/testFile656/beijing_GOT_faces_label_caffe/"
    fileRes = r'D:/face_recognition/testFile656/beijing_GOT_faces_caffe.txt'

    excel = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = excel.add_sheet('sheet1', cell_overwrite_ok=True)
    cnt = 1
    faces = os.listdir(inputPath)
    for face in faces:
        facePath = inputPath+face
        img = cv2.imread(facePath)

        #cv2.imshow("test",img)
        #cv2.waitKey(2000)

        minkey,minscore = img2score(img)
        with open(fileRes,'a+') as fr:
            fr.write('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore)+'\r')

        print ('No. '+ str(cnt) +' anskey = '+str(minkey)+'  minscore = '+str(minscore))
        sheet.write(cnt-1, 0, 'cnt')
        sheet.write(cnt-1, 1, cnt)
        sheet.write(cnt-1, 2, 'minkey')
        sheet.write(cnt-1, 3, minkey)
        sheet.write(cnt-1, 4, 'score')
        sheet.write(cnt-1, 5, minscore)
        shutil.copyfile(facePath, labelPath+minkey+face)
        print cnt
        cnt+=1
    excel.save(r'D:/face_recognition/testFile656/beijing_GOT_faces_caffe.xls')

def img2score(img): # only for one face
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    dets = detector(img)
    if(len(dets)==0):
        minkey = "Stranger_no_face"
        minscore = -1000
        #return
    else:
        for k, d in enumerate(dets):
            shape = predictor(gray, d)
            feat_pca = np.dot(getallfeat(shape,img)-mean,compoT)
            minkey,minscore=getscore(A,G,feat_pca,db_qf)
            break # one face default

    return minkey,minscore
def det2face(img, d):
    [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
    w = x2-x1
    h = y2-y1
    x1 = x1-w
    x2 = x2+w
    y1 = y1-h
    y2 = y2+h
    if x1<0:
      x1 = 10
    if y1<0:
      y1 = 10
    return img[y1:y2,x1:x2,:]
def getVideoFace():
    print "Begin!!!"
    videosPath = 'D:/face_recognition/testFile656/beijing_GOT_videos/'
    facesPath = 'D:/face_recognition/testFile656/beijing_GOT_faces/'
    #cnt = 1
    cnt = 1416
    cntV = 1
    videos = os.listdir(videosPath)
    print videos
    for video in videos:
        print videosPath+video
        capture = cv2.VideoCapture(videosPath+video)
        while capture.read():
            for i in range(7):
                ret,img=capture.read()
            #print type(img)
            if isinstance(img, types.NoneType):
                print ('Not a image. This is the end of the last vedio.')
                print ('Now lets begin the next!')
                break #This is the end of the vedio.
            dets = detector(img)
            for k, d in enumerate(dets):
                imgFace = det2face(img, d)
                imgFace = cv2.resize(imgFace,(300,300))
                cv2.imwrite(facesPath+str(cnt)+'.jpg',imgFace)
                print cnt
                cnt+=1
        print ("the {}-th video has finished".format(cntV))
        cntV+=1

    print "End!"

if __name__ == "__main__":
    caffeClassFaces()