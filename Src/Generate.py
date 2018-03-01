import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
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
#import cv2.cv as cv
import math
from scipy.io import savemat
from joint_bayesian import *
from sklearn.preprocessing import normalize
from sklearn.lda import LDA
import os
import common
from sklearn.externals import joblib
import ConfigParser
from skimage import transform as tf
from skimage.transform import warp
import copy
import multiprocessing

import math
import gc


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
        caffe.set_device(gpuid)
        
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
        caffe.set_device(0)
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
    


cf=ConfigParser.ConfigParser()
cf.read('config.ini')

#import caffe
caffe_root = cf.get("Parameters","caffe_root")
sys.path.insert(0, caffe_root + 'python')
import caffe

#qf_QueueSize=2
QMaxSize=4
videoQueue=multiprocessing.Queue(QMaxSize)
resultQueue=multiprocessing.Queue(1)
FaceImageQueue=multiprocessing.Queue()
ageSexResult=multiprocessing.Queue(10)

detect_count=0

pre_dets=set()

rec_threshold=int(cf.get("RecParam","threshold"))


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

def init_recog():
    f = open('../Models/db_20170425_lab_qf.pickle','rb')
    db_qf=pickle.load(f)
    f.close()

    count=0
   
    #####result######
    anspath='../Results/'
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    #out = cv2.VideoWriter(anspath+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi', fourcc, 12, (800,600))
    out=0
    #out = cv2.VideoWriter(anspath+filename, fourcc, 8, (800,600))
    ###############
    A,G=get_A_G_testset_feature()
    clt_pca = joblib.load('../Jb/'+"pca_model.m")
    return db_qf,count,out,A,G,clt_pca

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
    
def feat_norm(feat):
    tmp=np.zeros(feat.shape)
    for i in range(tmp.shape[0]):
        tmp[i]=feat[i]**2
    repfeat=feat/np.sqrt(sum(tmp.transpose()))
    return repfeat

def getallfeat(shape,img):
    #start=time.clock()
    imgs=normSingle(shape,img)
    #print "align time: ",time.clock()-start
    #start=time.clock()
    feat_wholeface=getfeat_github(net_wholeface,imgs[0])
    feat_ctf=getfeat_github(net_ctf,imgs[1])
    feat_le=getfeat_github(net_le,imgs[2])
    feat_re=getfeat_github(net_re,imgs[3])
    feat_eye=getfeat_github(net_eye,imgs[4])
    feat_mouth=getfeat_github(net_mouth,imgs[5])
    feat_downmouth=getfeat_github(net_downmouth,imgs[6])
    #print "get_featur time: ",time.clock()-start
    #start=time.clock()
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    #print "concatenate time: ",time.clock()-start
    return feat

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
    #print "pre_process time: ",time.clock()-start
    #start=time.clock()
    net_fused.blobs['data'].data[...]=INPUT
    out = net_fused.forward()
    #print "get_featur time: ",time.clock()-start
    #start=time.clock()
    feat_wholeface=feat_norm(net_fused.blobs['wholeface/fc5'].data[0].reshape(1,256))
    feat_ctf=feat_norm(net_fused.blobs['ctf/fc5'].data[0].reshape(1,256))
    feat_le=feat_norm(net_fused.blobs['le/fc5'].data[0].reshape(1,256))
    feat_re=feat_norm(net_fused.blobs['re/fc5'].data[0].reshape(1,256))
    feat_eye=feat_norm(net_fused.blobs['eye/fc5'].data[0].reshape(1,256))
    feat_mouth=feat_norm(net_fused.blobs['mouth/fc5'].data[0].reshape(1,256))
    feat_downmouth=feat_norm(net_fused.blobs['downmouth/fc5'].data[0].reshape(1,256))
    feat=np.concatenate((feat_wholeface,feat_ctf,feat_le,feat_re,feat_eye,feat_mouth,feat_downmouth),axis=1)
    #print "norm time: ",time.clock()-start
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
            if key.find('_')==-1:
                anskey=key
            else:
                anskey=key.split('_')[0]
    if(maxscore<rec_threshold):
        anskey='Stranger'
    return anskey,maxscore
	
def get_ratios(A, G, F1, F2):
    ratio = Verify(A, G, F1, F2)
    return ratio

def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(np.dot(np.transpose(x1), G), x2)
    return float(ratio)

def drawrec(img,d):
    color=(0,0,255)
    cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),color,2)
    


def create_database():
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
            feat=getallfeat(shape,im)
        database[file.split('.')[0]]=clt_pca.transform(feat)
        cnt=cnt+1

    f = open('../Models/db_20170425_lab_qf.pickle','wb')
    pickle.dump(database,f,-1)
    f.close()
    print cnt
    
	
def save_show_database(dict):
    detector = dlib.get_frontal_face_detector()
    path = "../Database" 
    files= os.listdir(path)
    cnt=0
    for file in files:
        if file.find('.')==-1:
            continue
        if not file.find('_')==-1:
            continue
        if(file.split('.')[1][0]=='p'):
            ppath=path+'/'+file.split('.')[0]+'.png'
        elif(file.split('.')[1][0]=='j'):
            ppath=path+'/'+file.split('.')[0]+'.jpg'
        else:
		    continue

        im=cv2.imread(ppath)
        gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        dets = detector(gray,1)
        if(len(dets)==0):
            continue
        for k, d in enumerate(dets):
            sr=im[d.top():d.bottom(),d.left():d.right(),:]
            sr=cv2.resize(sr,(80,80))
            dict[file.split('.')[0]]=sr

def InROI(rect):
    #print Roi_right,Roi_left,Roi_bottom,Roi_up
    #return True
    Roi_up=Pos_Mouse[0]
    Roi_left=Pos_Mouse[1]
    img_x=Pos_Mouse[2]
    img_y=Pos_Mouse[3]
    Roi_bottom=Pos_Mouse[4]
    Roi_right=Pos_Mouse[5]
    if(Roi_right-Roi_left<20 or Roi_bottom-Roi_up<20):
        return True
    if(Roi_left<img_x or Roi_up<img_y):
        return True
    
    if(rect.bottom()*ScaleY.value<Roi_up-img_y):
        return False
    if(rect.top()*ScaleY.value>Roi_bottom-img_y):
        return False
    if(rect.left()*ScaleX.value>Roi_right-img_x):
        return False
    if(rect.right()*ScaleX.value<Roi_left-img_x):
        return False
    return True

def face_add(set,rect):
    
    for item in set:
        #print item.left(),item.right(),item.top(),item.bottom()
        
        itemW=(item.right()-item.left())/2
        itemH=(item.bottom()-item.top())/2
        
        iCx=(item.right()+item.left())/2
        iCy=(item.bottom()+item.top())/2
        
        rectW=(rect.right()-rect.left())/2
        rectH=(rect.bottom()-rect.top())/2

        iRx=(rect.right()+rect.left())/2
        iRy=(rect.bottom()+rect.top())/2
        
        Cx=abs(iCx-iRx)
        Cy=abs(iCy-iRy)
        if(rectW+itemW<Cx or rectH+itemH<Cy):
            continue
        else:
            #print item.left(),item.right(),item.top(),item.bottom()
            #print rect.left(),rect.right(),rect.top(),rect.bottom()
            return

    set.add(rect)
    return
	
def face_dect(gray,detector):
    global pre_dets
    global detect_count
    if(len(pre_dets)==0 or detect_count%10==0):
        dets = detector(gray,1)
        pre_dets=copy.deepcopy(dets)
        detect_count=detect_count+1
        if detect_count>1000:
           detect_count=0
        return pre_dets
    tmp_dets=set()
    for k, d in enumerate(pre_dets):
        candi_left=max(d.left()-(d.right()-d.left()),0)
        candi_top=max(d.top()-(d.bottom()-d.top()),0)
        candi_right=min(d.right()+(d.right()-d.left()),gray.shape[1])
        candi_bottom=min(d.bottom()+(d.bottom()-d.top()),gray.shape[0])
        candi_img=gray[candi_top:candi_bottom,candi_left:candi_right].copy()
        cur_dets = detector(candi_img)
        for c,cur_d in enumerate(cur_dets):
            new_left=cur_d.left()+candi_left
            new_right=cur_d.right()+candi_left
            new_top=cur_d.top()+candi_top
            new_bottom=cur_d.bottom()+candi_top
            new_rec=dlib.rectangle(new_left,new_top,new_right,new_bottom)
            try:
                face_add(tmp_dets,new_rec)
                #tmp_dets.add(new_rec)
            except Exception as e:
                print("WTF", e)
    pre_dets=tmp_dets
    detect_count=detect_count+1
    if detect_count>1000:
       detect_count=0
    return pre_dets

def mtcnn_detect_face_draw(img,detector):
    #gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    total_boxes,points,numbox = detector.detectface(img)
    dets=set()
    for i in range(numbox):
        dets.add(dlib.rectangle(int(total_boxes[i][0]),int(total_boxes[i][1]),int(total_boxes[i][2]),int(total_boxes[i][3])))   
    if(len(dets)==0):
        #print "No face detected"
        return False
    rawimge=copy.deepcopy(img)
    #out.write(rawimge)
    if FaceImageQueue.qsize()>5:
        #FaceImageQueue.get(1,5)
        FaceImageQueue.put(rawimge)
    else:
        FaceImageQueue.put(rawimge)
    for k, d in enumerate(dets):
        if(InROI(d)):
            drawrec(img,d)
    if resultQueue.empty():
	    resultQueue.put([rawimge,dets])     

def detect_face_draw(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    detector = dlib.get_frontal_face_detector()
    #start=time.clock()
    #dets = detector(gray,1)
    dets=face_dect(gray,detector)
    #print "detect time: ",time.clock()-start

    if(len(dets)==0):
        print "No face detected"
        return False
    rawimge=copy.deepcopy(img)
    if FaceImageQueue.qsize()>5:
        FaceImageQueue.get(1,5)
        FaceImageQueue.put(rawimge)
    else:
        FaceImageQueue.put(rawimge)
    for k, d in enumerate(dets):
        if(InROI(d)):
            drawrec(img,d)
    if resultQueue.empty():
	    resultQueue.put([rawimge,dets])
def face_front_flag(gray):
    #gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #1
# Detect faces in the image
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5,5),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    ) #4

    if(len(faces) == 0):
        return False
    return True

def face_recog(img,dets):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    for k, d in enumerate(dets):
       
            shape = predictor(img, d)
            feat_pca=clt_pca.transform(getallfeat(shape,img))
            minkey,minscore=getscore(A,G,feat_pca,db_qf)
####################################################
            angle_flag = face_front_flag(candi_img)
#######################################################
            sevens=np.zeros((6,2))
            index=[36,39,42,45,48,54]

            for i in range(len(index)):
                sevens[i][0] = shape.part(index[i]).x
                sevens[i][1] = shape.part(index[i]).y
            ctf_index= np.array([[78,100],[95,100],[129,100],[146,100],[87,150],[138,150]])
            oriimg=sevens
            mapping=tf.estimate_transform('similarity',oriimg ,ctf_index)
            alignimg=warp(color_candi_img, inverse_map=mapping.inverse,output_shape=(224,224))


        #Gender prediction
            face_attr_net.blobs['data'].data[...] = transformer.preprocess('data',alignimg)
            out = face_attr_net.forward()
            prob = face_attr_net.blobs['prob'].data[0].flatten()
        #Age prediction
            prob_age = face_attr_net.blobs['prob-age'].data[0].flatten()
            age_estimation = 0
        #Smile prediction
            prob_smile = face_attr_net.blobs['prob-smile'].data[0].flatten()
        #age computing

            for i,j in enumerate(prob_age) :
                age_estimation += (i * j)

            if age_estimation >= 75 :
                show_age = 'Age: ' +  '80+'
            elif age_estimation < 5 :
                show_age = 'Age: ' +  '5-'
            else :
                show_age = 'Age: ' + str(round(age_estimation , 2))

            s_index = prob_smile.argsort()[-1] 
            s_proba = prob_smile[s_index]        
            show_smile = ''.join(['Smile: ',smile_list[s_index]]) 

        ###################
            g_index = prob.argsort()[-1] 
            g_proba = prob[g_index]        
            show_gender = gender_list[g_index]
            #print show_age,show_smile,show_gender
######################################################################
            try:
                sr=img[d.top():d.bottom(),d.left():d.right(),:]
                sr=cv2.resize(sr,(80,80))
                #if (not ageSexResult.full()) and Showcount.value%2==0 and angle_flag and size_flag:#if not ageSexResult.full():
                if angle_flag and size_flag and not ageSexResult.full():
	                ageSexResult.put([sr,show_age,show_gender])     
                if not minkey=='Stranger':
                    SmallFace[minkey]=[sr,show_age,show_smile,show_gender,angle_flag,size_flag]
 
            except:
                pass
    return True
#####################################################
def face_attr_estimation(im,dets,transformer):
        #Labels
    gender_list=['Female','Male']
    smile_list = ['NO','YES']
    for k, d in enumerate(dets):
        [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]

        shape = predictor(im, d) 

        sevens=np.zeros((6,2))
        index=[36,39,42,45,48,54]

        for i in range(len(index)):
            sevens[i][0] = shape.part(index[i]).x
            sevens[i][1] = shape.part(index[i]).y
        ctf_index= np.array([[78,100],[95,100],[129,100],[146,100],[87,150],[138,150]])
        oriimg=sevens
        mapping=tf.estimate_transform('similarity',oriimg ,ctf_index)
        alignimg=warp(im, inverse_map=mapping.inverse,output_shape=(224,224))
        #Gender prediction
        face_attr_net.blobs['data'].data[...] = transformer.preprocess('data',alignimg)
        out = face_attr_net.forward()
        prob = face_attr_net.blobs['prob'].data[0].flatten()
        #Age prediction
        prob_age = face_attr_net.blobs['prob-age'].data[0].flatten()
        age_estimation = 0
        #Smile prediction
        prob_smile = face_attr_net.blobs['prob-smile'].data[0].flatten()
        #age computing

        for i,j in enumerate(prob_age) :
            age_estimation += (i * j)

        if age_estimation >= 75 :
            show_age = 'Age: ' +  '80+'
        elif age_estimation < 5 :
            show_age = 'Age: ' +  '5-'
        else :
            show_age = 'Age: ' + str(round(age_estimation , 2))

        s_index = prob_smile.argsort()[-1] 
        s_proba = prob_smile[s_index]        
        show_smile = ''.join(['Smile: ',smile_list[s_index]]) 

        ###################
        g_index = prob.argsort()[-1] 
        g_proba = prob[g_index]        
        show_gender = gender_list[g_index]
	print show_age,show_smile,show_gender
    return [show_age,show_smile,show_gender]

#####################################################
class ReadVideo(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
    def run(self):
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
        if not inputtype==3:
            while True:
                if readPause.value:
                    ret,face_img = capture.read()
                    #print face_img.shape
                    if not videoQueue.full():
                        videoQueue.put(face_img)
        else:
            while True:
                if readPause.value:
                    if not videoQueue.full():
                        ret,face_img = capture.read()
                        
                        videoQueue.put(face_img)
                        
		
		
class FaceRec(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)

    def run(self):
        global db_qf,count,out,A,G,clt_pca
        db_qf,count,out,A,G,clt_pca=init_recog()
        global predictor
        predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
        
        predictor = dlib.shape_predictor(predictor_path)
        caffe.set_mode_gpu()
        caffe.set_device(0)
        
        #global net_fused
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
####################################################################
        global face_attr_net
        face_attr_net = caffe.Net('../Models/face_attr_deploy.prototxt',
                    '../Models/face_attr.caffemodel',
                    caffe.TEST)
        transformer = caffe.io.Transformer({'data': face_attr_net.blobs['data'].data.shape})
        transformer.set_transpose('data',(2,0,1))
        mean_value = np.array([ 104.00698793,116.66876762,122.67891434])
        transformer.set_mean('data',mean_value)
        transformer.set_channel_swap('data',(0,1,2))
        transformer.set_raw_scale('data',255)
####################################################################
        
        while True:
            if not resultQueue.empty() and readPause.value:
                [face_img,dets]=resultQueue.get(1,5)
                flag=face_recog(face_img,dets,transformer)
                #face_attr_info = face_attr_estimation(face_img,dets,transformer)
            if not readPause.value and UpdateValue.value:
                create_database()
                f = open('../Models/db_20170425_lab_qf.pickle','rb')
                db_qf=pickle.load(f)
                f.close()
                save_show_database(ResDict)
                UpdateValue.value= not UpdateValue.value
                print "update Down"
'''         	
class UpdateDataBase(multiprocessing.Process):
    def __init__(self):
 
        multiprocessing.Process.__init__(self)

    def run(self):
        global db_qf,count,out,A,G,clt_pca
        db_qf,count,out,A,G,clt_pca=init_recog()
        global predictor,detector
        predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        caffe.set_mode_gpu()
        caffe.set_device(0)
        
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
        
        create_database()
        del net_ctf,net_downmouth,net_eye,net_le,net_mouth,net_re,net_wholeface
        del db_qf,count,out,A,G,clt_pca,predictor,detector
'''
		

class ShowR(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
    def run(self):
        global app
        app = QApplication(sys.argv)
        player=Timer("PlayVideo")
        Recplayer=RectTimer("ShowResult")
        ageSexPlayer=AgeSexTimer("ShowAgeSexResult")
        main = MainWindow(player,Recplayer,ageSexPlayer)
        main.show()
        app.installEventFilter(main)
        sys.exit(app.exec_())

		
class SaveImage(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
    def run(self):
        while True:
            if SaveImageValue.value and not FaceImageQueue.empty():
                im=FaceImageQueue.get(1,5)
                print im.shape
                cv2.imwrite('../SaveImage/'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.jpg',im)
                SaveImageValue.value=not SaveImageValue.value
                

class MainWindow(QWidget):
    def __init__(self,playtimer,Recplayer,ageSexPlayer,parent = None):
        QWidget.__init__(self)
        self.resize(550, 480)
        self.setWindowTitle('face recognition')
        self.status = 0 #0 is init status;1 is play video; 2 is capture video
        self.image = QImage()
        self.update = QPushButton('UpdateDB')
        self.playbtn = QPushButton('Start')
        self.SaveImage = QPushButton('Capture')
        exitbtn = QPushButton('exit')
        
        self.recimg=QImage()
        self.ageSeximg=QImage()
        
        
        vbox = QHBoxLayout()
        vbox.addWidget(self.update)
        vbox.addWidget(self.playbtn)
        vbox.addWidget(exitbtn)
        vbox.addWidget(self.SaveImage)
        
        self.piclabel = QLabel('')
        self.rectImg = QLabel('')
        self.ageSexImg = QLabel('')
		
        lbox=QVBoxLayout()
        lbox.addWidget(self.piclabel)
        lbox.addStretch(1)
        lbox.addWidget(self.ageSexImg)
    
        hbox = QHBoxLayout()
        hbox.addLayout(lbox)
        hbox.addStretch(1)
        hbox.addWidget(self.rectImg)
		
        fbox=QVBoxLayout()
        fbox.addLayout(vbox)
        fbox.addStretch(1)
        fbox.addLayout(hbox)
        #fbox.addStretch(1)
        #fbox.addWidget(self.ageSexImg)

        self.setLayout(fbox)

        self.playtimer = playtimer

        self.rectResult=Recplayer
        self.ageSex = ageSexPlayer

        self.update.clicked.connect(self.updateDB)

        self.connect(self.playtimer, SIGNAL("PlayVideo"), 
                                                    self.PlayVideo)  
        self.connect(self.rectResult, SIGNAL("ShowResult"), 
                                                    self.ShowResult)  
        self.connect(self.ageSex, SIGNAL("ShowAgeSexResult"), 
                                                    self.ShowAgeSexResult)  
        self.connect(self.playbtn, SIGNAL("clicked()"),
                                                     self.VideoPlayPause)
        self.connect(exitbtn, SIGNAL("clicked()"), 
                                              self.shutDown)
        #self.btnExit.clicked.connect(self.close)
        #self.actionExit.triggered.connect(self.close)
        self.connect(self.SaveImage, SIGNAL("clicked()"), 
                                              self.saveImage)
    def closeEvent(self, event):
        #print("event")
        ShutDownValue.value=not ShutDownValue.value
        event.accept()
        
    def PlayVideo(self,im):
        self.image.loadFromData(QByteArray(im))
        self.piclabel.setPixmap(QPixmap.fromImage(self.image))
        if not UpdateResult.value:
            UpdateResult.value=not UpdateResult.value

		
    def ShowResult(self,im):
        self.recimg.loadFromData(QByteArray(im))
        self.rectImg.setPixmap(QPixmap.fromImage(self.recimg))

		
    def ShowAgeSexResult(self,im):
        self.ageSeximg.loadFromData(QByteArray(im))
        self.ageSexImg.setPixmap(QPixmap.fromImage(self.ageSeximg))
		
		
    def VideoPlayPause(self):
        self.status, playstr, capturestr = ((1, 'pause', 'capture'), (0, 'play', 'capture'), (1, 'pause', 'capture'))[self.status]#
        self.playbtn.setText(playstr)
        
        if self.status is 1:
            self.playtimer.start()
            self.rectResult.start()
            self.ageSex.start()
            readPause.value = not readPause.value
            UpdateResult.value=not UpdateResult.value
        else:
            self.playtimer.stop()
            self.rectResult.stop()
            self.ageSex.stop()
            readPause.value = not readPause.value
            UpdateResult.value=not UpdateResult.value
    def updateDB(self):
        if not readPause.value:
            UpdateValue.value= not UpdateValue.value
    def shutDown(self):
        #print "success"
        ShutDownValue.value=not ShutDownValue.value
        
    def saveImage(self):
        SaveImageValue.value=not SaveImageValue.value
            
    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            pos = event.pos()
            #global Roi_up
            Pos_Mouse[0]=pos.y()
            #global Roi_left
            Pos_Mouse[1]=pos.x()
            #global img_x
            Pos_Mouse[2]=self.piclabel.pos().x()
            #global img_y
            Pos_Mouse[3]=self.piclabel.pos().y()

        if event.type()==QEvent.MouseButtonRelease:
            pos = event.pos()
            #global Roi_bottom
            Pos_Mouse[4]=pos.y()
            #global Roi_right
            Pos_Mouse[5]=pos.x()
        	
        return QMainWindow.eventFilter(self, source, event)	

        

class Timer(QThread):
    def __init__(self, signal = "updateTime()", parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        flag=True
        anspath='../Results/'
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
        
        mtcnn_detector = FaceDetector(minsize = 80, gpuid = 0, fastresize = False)

        while True:
            if self.stoped:
                return
            if not videoQueue.empty():
                resim=videoQueue.get(1,5)
                #detect_face_draw(resim)
                if flag:
                    ScaleX.value=800*1.0/resim.shape[0]
                    ScaleY.value=600*1.0/resim.shape[1]
                    flag=False
                    #out = cv2.VideoWriter(anspath+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi', fourcc, 12,(1920,1080))
                mtcnn_detect_face_draw(resim,mtcnn_detector)
                #out.write(resim)
                resim=cv2.resize(resim,(800,600))
                #out.write(resim)
                #resim=cv2.resize(resim,(int(resim.shape[1]*ShowScale.value),int(resim.shape[0]*ShowScale.value)))
                cv2_im = cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)
                im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
                self.emit(SIGNAL(self.signal),im)
                global Showcount
                Showcount.value=Showcount.value+1
                if Showcount.value>100000:
                    Showcount.value=0
                #time.sleep(0.05)
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True
        
    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped


class RectTimer(QThread):
    def __init__(self, signal = "updateTime()", parent=None):
        super(RectTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        fontsize=0.8
        font=cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX,0.1,fontsize,0,1,cv.CV_AA)
        color=(0,0,255)
        timedict={}
        showdict={}
        age_dict={}
        while True:
            if self.stoped:
                return
            if UpdateResult.value:
                UpdateResult.value=not UpdateResult.value
            else:
                continue
            resim=np.zeros((760,200,3),dtype=np.uint8)
            resim[:,:,:]=180
            count=0
            for key in SmallFace.keys():
                showdict[key]=SmallFace[key][0]
                if SmallFace[key][4] and SmallFace[key][5]:
                    age_dict[key]=SmallFace[key][1:4]
                timedict[key]=Showcount.value
                SmallFace.pop(key, None)
            #print "check"
				
            for key in showdict.keys():
                if(Showcount.value-timedict[key]>100):
                    continue
                try:
                    resim[count*120+40:(count+1)*120,10:90,:]=showdict[key]
                    resim[count*120+40:(count+1)*120,110:190,:]=ResDict[key]
                except:
                    pass
                cv.PutText(cv.fromarray(resim),"Name: "+key,(0,count*120+15),font,color)
                try:
                    cv.PutText(cv.fromarray(resim),age_dict[key][0],(0,count*120+35),font,color)
                    cv.PutText(cv.fromarray(resim),age_dict[key][1],(80,count*120+35),font,color)
                    cv.PutText(cv.fromarray(resim),age_dict[key][2],(160,count*120+35),font,color)
                except:
                    cv.PutText(cv.fromarray(resim),"Age: ",(0,count*120+35),font,color)
                    cv.PutText(cv.fromarray(resim),"Smile: ",(80,count*120+35),font,color)
                    cv.PutText(cv.fromarray(resim),"NO",(160,count*120+35),font,color)
                count=count+1
                
            cv2_im = cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
            self.emit(SIGNAL(self.signal),im)
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True
        
    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped
			
class AgeSexTimer(QThread):
    def __init__(self, signal = "updateTime()", parent=None):
        super(AgeSexTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        fontsize=1
        font=cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX,0.1,fontsize,0,1,cv.CV_AA)
        color=(0,0,255)
        age_wl=[]

        while True:
            if self.stoped:
                return
            if UpdateResult.value:
                UpdateResult.value=not UpdateResult.value
            else:
                continue
            resim=np.zeros((160,800,3),dtype=np.uint8)
            resim[:,:,:]=180
            count=0
          
            if not ageSexResult.empty() and Showcount.value%2==0:
                if(len(age_wl)<10):
                    age_wl.append(ageSexResult.get(1,5))
                else:
                    del age_wl[0]
                    age_wl.append(ageSexResult.get(1,5))
					
            if not ageSexResult.empty() and Showcount.value%2==0:
                if(len(age_wl)<10):
                    age_wl.append(ageSexResult.get(1,5))
                else:
                    del age_wl[0]
                    age_wl.append(ageSexResult.get(1,5))
				
            for item in age_wl:
                try:
                    resim[5:85,count*80:(count+1)*80,:]=item[0]
                except:
                    pass
                cv.PutText(cv.fromarray(resim),item[1].split(' ')[0],(count*80,100),font,color)
                cv.PutText(cv.fromarray(resim),item[1].split(' ')[1],(count*80,120),font,color)
                cv.PutText(cv.fromarray(resim),item[2],(count*80,140),font,color)
                count=count+1
                
            cv2_im = cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
            self.emit(SIGNAL(self.signal),im)
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True
        
    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped
def get_mac_address():
    import uuid
    node = uuid.getnode()
    mac = uuid.UUID(int = node).hex[-12:]
    return mac

if __name__ == "__main__" :

    caffe.set_device(0)
    caffe.set_mode_gpu()  # set gpu model
    global db_qf,count,out,A,G,clt_pca
    db_qf,count,out,A,G,clt_pca=init_recog()
    global predictor
    predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
        
    predictor = dlib.shape_predictor(predictor_path)
    global mtcnn_detector
    mtcnn_detector = FaceDetector(minsize = 80, gpuid = 0, fastresize = False)
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
    flist=open("ZSH_list.txt")
    oneline=flist.readline()
    data={}
    count=0
    while oneline:
        path='/home/sam/Desktop/train/'+oneline.split(' ')[0]
        print path
        img = cv2.imread(path)
        total_boxes,points,numbox = mtcnn_detector.detectface(img)
        dets=set()
        for i in range(numbox):
            dets.add(dlib.rectangle(int(total_boxes[i][0]),int(total_boxes[i][1]),int(total_boxes[i][2]),int(total_boxes[i][3])))   
            break

        if(len(dets)==0):
            oneline=flist.readline()
            continue
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            feat_pca=getallfeat(shape,img)
            data[oneline.split(' ')[0]]=feat_pca
        oneline=flist.readline()
        count=count+1
        if(count%10000==0):
            f = open('./Features/CASIA_features_%s.pickle'%count,'wb')
            pickle.dump(data,f,-1)
            f.close()
            data={}
            gc.collect()
    f = open('./Features/CASIA_features_%s.pickle'%count,'wb')
    pickle.dump(data,f,-1)
    f.close()
    data={}
    gc.collect()
    
            
