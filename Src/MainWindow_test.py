#!/usr/bin/python
# -*- coding: utf-8 -*-

# boxlayout.py

import sys
#from PyQt4 import QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import cv2
from PIL import Image
import multiprocessing
import time

videoQueue = multiprocessing.Queue(4)
class ReadVideo(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
        #self.videoQueue = multiprocessing.Queue(4)
    def run(self):
        videoPath = '../Testdata/20170614.avi'
        capture=cv2.VideoCapture(videoPath)
        #capture = cv2.VideoCapture('D:/face_recognition/Testdata/20170528.avi')
        while(True):
            #if not videoQueue.full():
            if True:
                ret,face_img = capture.read()
                cv2.imshow('test',face_img)
                cv2.waitKey(10)
                #global t
                #videoQueue.put(face_img)


def show(img):
    cv2.imshow('test',img)
    cv2.waitKey(10)


'''
class ShowVideo(multiprocessing.Process):
    def __init__(self,read):
        multiprocessing.Process.__init__(self)
        self.read = read
    def run(self):
        while(True):
            print self.read.videoQueue.empty()
            time.sleep(1)
'''

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.status = 0
        self.capturebtn = QPushButton('capture')
        self.playbtn = QPushButton("Play")
        self.cancelButton = QPushButton("Cancel")
        self.exitbtn = QPushButton('exit')
        self.image = QImage()
        self.imageLabel = QLabel('')
        self.faceLabel = QLabel('')
        self.idLabel = QLabel('')

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
        self.buttons.addWidget(self.playbtn)
        self.buttons.addWidget(self.cancelButton)
        self.buttons.addWidget(self.exitbtn)

        self.whole = QVBoxLayout()
        self.whole.addLayout(self.videoImage)
        self.whole.addLayout(self.buttons)

        self.setLayout(self.whole)

        self.showImageLabel('head.jpg')
        self.showFaceLabel('hou.bmp')
        self.showIdLabel('head.jpg')

        self.playtimer = Timer("updatePlay")
        self.connect(self.playtimer, SIGNAL("updatePlay"), self.PlayVideo)  

        self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        self.connect(self.playbtn, SIGNAL("clicked()"), self.VideoPlayPause)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self.exit)
        self.setWindowTitle('Face Recognition')
        self.resize(1170, 700)

    def PlayVideo(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        
    def VideoPlayPause(self):
        self.status, playstr, capturestr = ((1, 'pause', 'capture'), (0, 'play', 'capture'), (1, 'pause', 'capture'))[self.status]#三种状态分别对应的显示、处理
        self.playbtn.setText(playstr)
        self.capturebtn.setText(capturestr)
        if self.status is 1:
            #self.timer.stop()
            self.playtimer.start()
        else:
            self.playtimer.stop()    

    def showImageLabel(self,im):
        orig_im = cv2.imread(im)
        resize_im=cv2.resize(orig_im,(900,630))
        cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        self.imageLabel.setScaledContents(True)
    def showFaceLabel(self,im):
        orig_im = cv2.imread(im)
        resize_im=cv2.resize(orig_im,(220,300))
        cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
        self.image.loadFromData(QByteArray(byte_im))
        self.faceLabel.setPixmap(QPixmap.fromImage(self.image))
        self.faceLabel.setScaledContents(True)
    def showIdLabel(self,im):
        orig_im = cv2.imread(im)
        resize_im=cv2.resize(orig_im,(220,300))
        cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
        self.image.loadFromData(QByteArray(byte_im))
        self.idLabel.setPixmap(QPixmap.fromImage(self.image))
        self.idLabel.setScaledContents(True)
    '''
    def exitself(self):
        global t
        print id(t)
    '''
class Timer(QThread):
    
    def __init__(self, signal = "updateTime", parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()


    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        camera = cv2.VideoCapture('../Testdata/20170614.avi')
        #camera = cv2.VideoCapture(1)
        
        while True:
            if self.stoped:
                return

            ret, face = camera.read()
            resize_im=cv2.resize(face,(220,300))
            cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
            self.emit(SIGNAL(self.signal),byte_im)
            time.sleep(0.04) #40毫秒发送一次信号，每秒25帧
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped
'''
class MainFun(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
    def run(self):
        app = QApplication(sys.argv)
        mainwindow = MainWindow()
        mainwindow.show()
        sys.exit(app.exec_())
'''

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec_())
'''
if __name__ == '__main__':
    #global t
    #t = multiprocessing.Value("i",0)
    mainFun = MainFun()
    mainFun.start()
    #readVideo = ReadVideo()
    #showVideo = ShowVideo(readVideo.videoQueue)
    #readVideo.start()
    #showVideo.start()
'''



    #while True:
    #    if not readVideo.videoQueue.empty():
    #        cv2.imshow("test",readVideo.videoQueue.get(1,5))
    #        cv2.waitKey(10)