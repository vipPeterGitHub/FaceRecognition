#coding=utf-8
from face_recognition_new import *
caffe.set_mode_gpu()

#capture = cv2.VideoCapture('../Testdata/20170528.avi')
#capture = cv2.VideoCapture('../Testdata/2018_03_12_16_03_18.avi')
capture = cv2.VideoCapture(0)

def convImg(img,w,h):
    resize_im=cv2.resize(img,(w,h))
    cv2_im = cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    byte_im = pil_im.convert('RGB').tobytes('jpeg', 'RGB')
    return byte_im

def readIDcard():
    image_path = 'D:/IDcard_test_program/'
    #history_path = 'D:/testImage/history/'
    image_name = os.listdir(image_path)
    if image_name.__len__()==14:
        begin = time.clock()
        pic_path = image_path + image_name[13]
        id_num = pic_path.split('/')[2].split('.')[0].split('_')[1]
        name = pic_path.split('/')[2].split('.')[0].split('_')[0]
        name = name.decode("gb2312")
        pic_path_new = 'D:/IDcard_test_program/'+id_num+'.jpg'
        shutil.move(pic_path, pic_path_new)
        img = cv2.imread(pic_path_new)
        byte_im = convImg(img,220,300)
        picklePath = '../Models/'+id_num+'_DB.pickle'
        createDatabase(id_num,pic_path_new,picklePath)
        end = time.clock()
        print ("Create Database time is {}".format(end - begin))
        '''
        begin = time.clock()
        compareFace(cv2.imread('hou1c.jpg'),picklePath)
        end = time.clock()
        print ("compare time is {}".format(end - begin))
        '''
        os.remove(pic_path_new)
    else:
        id_num = 0
        byte_im = 0
        picklePath = 0
        name = 0
    return id_num, byte_im, picklePath, name


class MainWindow(QWidget):
    def __init__(self, qFace, qDet, qGray, qImg, qImgFace, qFeat, qFeat_Img):
        super(MainWindow, self).__init__()
        self.status = 0
        self.playButton = QPushButton("Play")
        self.playButton.setStyleSheet('font-size:30px; border-radius:10px;border:3px groove gray')#;
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setStyleSheet('font-size:30px')
        self.exitbtn = QPushButton('Exit')
        self.exitbtn.setStyleSheet('font-size:30px')
        self.exitbtn.setFixedWidth(300)  
        self.exitbtn.setFixedHeight(50)
        self.playButton.setFixedWidth(300)  
        self.playButton.setFixedHeight(50)
        self.cancelButton.setFixedWidth(300)  
        self.cancelButton.setFixedHeight(50)

        self.image = QImage()
        self.imageLabel = QLabel('')
        self.faceLabel = QLabel('')
        self.idLabel = QLabel('')
        self.face1 = QLabel('')
        self.face2 = QLabel('')
        '''
        self.textLabel = QLabel("Hello World!")
        self.textLabel.setFixedWidth(1170)  
        self.textLabel.setFixedHeight(100)  
        self.textLabel.setAlignment(Qt.AlignCenter)
        self.textLabel.setFont(QFont("Roman times",50,QFont.Bold))
        '''


        ######## multi_face_show ############
        
        self.face11 = QLabel('')
        self.face12 = QLabel('')
        self.text1 = QLabel("Stranger")
        self.setTextFont(self.text1, w = 250, h = 30, s = 15)
        self.face21 = QLabel('')
        self.face22 = QLabel('')
        self.text2 = QLabel("Stranger")
        self.setTextFont(self.text2, w = 250, h = 30, s = 15)
        self.face31 = QLabel('')
        self.face32 = QLabel('')
        self.text3 = QLabel("Stranger")
        self.setTextFont(self.text3, w = 250, h = 30, s = 15)
        self.face41 = QLabel('')
        self.face42 = QLabel('')
        self.text4 = QLabel("Stranger")
        self.setTextFont(self.text4, w = 250, h = 30, s = 15)
        '''
        self.f1 = QHBoxLayout()
        self.f1.addWidget(self.face11)
        self.f1.addWidget(self.face12)
        self.f2 = QHBoxLayout()
        self.f2.addWidget(self.face21)
        self.f2.addWidget(self.face22)
        self.f3 = QHBoxLayout()
        self.f3.addWidget(self.face31)
        self.f3.addWidget(self.face32)
        self.f4 = QHBoxLayout()
        self.f4.addWidget(self.face41)
        self.f4.addWidget(self.face42)
        self.side = QVBoxLayout()
        self.side.addLayout(self.f1)
        self.side.addWidget(self.text1)
        self.side.addLayout(self.f2)
        self.side.addWidget(self.text2)
        self.side.addLayout(self.f3)
        self.side.addWidget(self.text3)
        self.side.addLayout(self.f4)
        self.side.addWidget(self.text4)

        self.setDefaultPic(self.face11,120,120)
        self.setDefaultPic(self.face12,120,120)
        self.setDefaultPic(self.face21,120,120)
        self.setDefaultPic(self.face22,120,120)
        self.setDefaultPic(self.face31,120,120)
        self.setDefaultPic(self.face32,120,120)
        self.setDefaultPic(self.face41,120,120)
        self.setDefaultPic(self.face42,120,120)
        '''

        ######## multi_face_show ############


        self.textLabel2 = QLabel("Click Play to Start!")
        '''
        myname = '\xba\xf2\xb1\xcc\xcc\xb6'
        s1 = myname.decode("gb2312")
        self.textLabel2 = QLabel(s1)
        '''
        self.setTextFont(self.textLabel2, w = 1100, h = 100, s = 35)

        self.twoFace2 = QVBoxLayout()
        self.twoFace2.addWidget(self.face1)
        self.twoFace2.addWidget(self.face2)

        self.twoFace = QVBoxLayout()
        self.twoFace.addWidget(self.faceLabel)
        #self.twoFace.addStretch(1)
        self.twoFace.addWidget(self.idLabel)

        self.videoImage = QHBoxLayout()
        self.videoImage.addWidget(self.imageLabel)
        #self.videoImage.addLayout(self.twoFace)
        self.videoImage.addLayout(self.twoFace2)
        #self.videoImage.addLayout(self.side)
        

        self.buttons = QHBoxLayout()
        #hbox.addStretch(1)
        self.buttons.addWidget(self.playButton)
        self.buttons.addWidget(self.cancelButton)
        self.buttons.addWidget(self.exitbtn)

        self.whole = QVBoxLayout()
        self.whole.addLayout(self.videoImage)
        #self.whole.addWidget(self.textLabel)
        self.whole.addWidget(self.textLabel2)
        self.whole.addLayout(self.buttons)
        
        self.setLayout(self.whole)

        '''
        self.setDefaultPic(self.imageLabel,900,630,'nopic.jpg')
        self.setDefaultPic(self.faceLabel)
        self.setDefaultPic(self.idLabel)
        self.setDefaultPic(self.face1)
        self.setDefaultPic(self.face2)
        self.face2.setText(" ")
        '''

        self.qFace = qFace
        self.qDet = qDet
        self.qGray = qGray
        self.qImg = qImg
        self.qImgFace = qImgFace
        self.qFeat = qFeat
        self.qFeat_Img = qFeat_Img
        self.playtimer = Timer("videoPlay", self.qFace)
        self.facetimer = FaceTimer(self.qFace, "facePlay", self.qDet, self.qGray, self.qImg, self.qImgFace)
        self.idcardtimer = IdcardTimer(self.qFace, "idcardPlay")
        self.texttimer = TextTimer(self.qDet, self.qGray, self.qImg, self.qImgFace, "textPlay", self.qFeat, self.qFeat_Img)

        self.connect(self.texttimer, SIGNAL("textPlay"), self.multiFace)
        #self.connect(self.texttimer, SIGNAL("textPlay"), self.playText)

        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)
        self.connect(self.facetimer, SIGNAL("facePlay"), self.playFace)
        self.connect(self.idcardtimer, SIGNAL("idcardPlay"), self.playIdcard)
        self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        self.connect(self.playButton, SIGNAL("clicked()"), self.VideoPlayPause)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self.exit)
        self.setWindowTitle('Face Recognition')
        #self.resize(1170, 800)
        self.resize(1100, 830)
        self.cnt = 0
        self.name = ["Stranger","Stranger","Stranger","Stranger"]
    def multiFace(self):#, im, db, name):
        pass
        '''
    	if name != "Stranger": #and name != self.name[0] and name != self.name[1] and name != self.name[2] and name != self.name[3]:
    		if self.cnt == 0 :#and name != self.name[3]:
    			self.showResult(self.face11, self.face12, self.text1, im, db, name)
    			self.name[0] = name
    			self.cnt += 1
    		elif self.cnt == 1 :#and name != self.name[0]:
    			self.showResult(self.face21, self.face22, self.text2, im, db, name)
    			self.name[1] = name
    			self.cnt += 1
    		elif self.cnt == 2 :#and name != self.name[1]:
    			self.showResult(self.face31, self.face32, self.text3, im, db, name)
    			self.name[2] = name
    			self.cnt += 1
    		elif self.cnt == 3 :#and name != self.name[2]:
    			self.showResult(self.face41, self.face42, self.text4, im, db, name)
    			self.name[3] = name
    			self.cnt = 0
        '''
    '''
    def showResult(self, imLabel1, imLabel2, tLabel, im, db, name):
        self.image.loadFromData(QByteArray(im))
        imLabel1.setPixmap(QPixmap.fromImage(self.image))
        self.image.loadFromData(QByteArray(db))
        imLabel2.setPixmap(QPixmap.fromImage(self.image))
        tLabel.setText(name)
    '''
    '''
    def playText(self, s, byte_im, db_im_s):
        #print s
        self.textLabel.setText(s)
        self.image.loadFromData(QByteArray(byte_im))
        self.face1.setPixmap(QPixmap.fromImage(self.image))
        db_im = cv2.imread(db_im_s)
        byte_im_db = convImg(db_im,220,300)
        self.image.loadFromData(QByteArray(byte_im_db))
        self.face2.setPixmap(QPixmap.fromImage(self.image))
    '''
    def playVideo(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
    def playFace(self):#,byte_im):
        pass
        #self.image.loadFromData(QByteArray(byte_im))
        #self.faceLabel.setPixmap(QPixmap.fromImage(self.image))
    def playIdcard(self,s,id_byte_im,cap_byte_im):
    	self.textLabel2.setText(s)
        #self.textLabel2.setText(QString(s+u"中文"))
        if id_byte_im !=0:
            self.image.loadFromData(QByteArray(id_byte_im))
            self.face2.setPixmap(QPixmap.fromImage(self.image))
        else:
            self.face2.setText(" ")
        if cap_byte_im !=0:
            self.image.loadFromData(QByteArray(cap_byte_im))
            self.face1.setPixmap(QPixmap.fromImage(self.image))
        else:
            self.face1.setText(" ")
        #time.sleep(1)
    def VideoPlayPause(self):
        #self.status, playstr, capturestr = ((1, 'pause', 'capture'), (0, 'play', 'capture'), (1, 'pause', 'capture'))[self.status]#三种状态分别对应的显示、处理
        self.status, playstr = ((1, 'Pause'), (0, 'Play'))[self.status]
        self.playButton.setText(playstr)
        if self.status is 1:
            #self.timer.stop()
            self.playtimer.start()
            #self.facetimer.start()
            self.idcardtimer.start()
            #self.texttimer.start()
        else:
            self.playtimer.stop()
            #self.facetimer.stop()
            self.idcardtimer.stop()
            #self.texttimer.stop()

    def setTextFont(self,l, w = 250, h = 30, s = 8):
    	l.setFixedWidth(w)  
        l.setFixedHeight(h)  
        l.setAlignment(Qt.AlignCenter)
        #l.setFont(QFont("Roman times",s,QFont.Bold))
        #l.setFont(QFont("Arial",s,QFont.Normal))
        l.setFont(QFont("Microsoft YaHei",s,QFont.Normal))
    def setDefaultPic(self,l,w = 220,h = 300,im = 'nopic.jpg'):
    	orig_im = cv2.imread(im)
        byte_im = convImg(orig_im,w,h)
        self.image.loadFromData(QByteArray(byte_im))
        l.setPixmap(QPixmap.fromImage(self.image))
        #l.setScaledContents(True)

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
            begin = time.clock()
            ret, face = capture.read()
            #out.write(face)
            face = cv2.flip(face, 1)
            if not qFace.full():
                qFace.put(face)
            else:
                qFace.queue.clear()
                qFace.put(face)
            byte_im = convImg(face,900,630)
            self.emit(SIGNAL(self.signal),byte_im)
            time.sleep(0.04)
            end = time.clock()
            print("capture time is {}".format(end - begin))
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class FaceTimer(QThread):
    
    def __init__(self, qFace = Queue.Queue(4), signal = "updateTime", 
                        qDet = Queue.Queue(4), qGray = Queue.Queue(4), 
                        qImg = Queue.Queue(4), qImgFace = Queue.Queue(4), 
                        parent=None):
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
                    imgFace = cv2.imread("nopic.jpg")
                else:
                    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    dets = detector(gray)
                    if(len(dets)==0):
                        print "No face detected"
                        minkey = 'Stranger'
                        imgFace = cv2.imread("nopic.jpg")
                    else:
                        for k, d in enumerate(dets):
                            [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                            if x1<0:
                                x1 = 10
                            #print ("img_shape = {}".format(img.shape))
                            imgFace = img[y1:y2,x1:x2,:]
                            if not qDet.full():
                                qDet.put(d)
                                qGray.put(gray)
                                qImg.put(img)
                            	qImgFace.put(imgFace)
                            #print ("x1={}x2={}y1{}y2{}".format(x1,x2,y1,y2))
                            #break
                byte_im = convImg(imgFace,220,300)
                self.emit(SIGNAL(self.signal))#,byte_im)
                time.sleep(0.04)
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class TextTimer(QThread):
    
    def __init__(self, qDet = Queue.Queue(4), 
                        qGray = Queue.Queue(4), qImg = Queue.Queue(4),
                        qImgFace = Queue.Queue(4), signal = "updateTime",
                        qFeat = Queue.Queue(4), qFeat_Img = Queue.Queue(4), parent=None):
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
                imgFace = qImgFace.get(True,3)
                begin = time.clock()
                shape = predictor(gray, d) #0.06s
                caffe.set_mode_gpu()
                #caffe.set_device(0)
                begin = time.clock()
                feat_pca = np.dot(getallfeat(shape,img)-mean,compoT) #4s
                end = time.clock()
                print ("feat_pca processing time is {}".format(end-begin))
                if not qFeat.full():
                    qFeat.put(feat_pca)
                    qFeat_Img.put(imgFace)
                minkey,minscore=getscore(A,G,feat_pca,db_qf)
                print ("minscore is {}, minkey is {}".format(minscore,minkey))
                db_im = cv2.imread('../Database/'+minkey+'.jpg')
                self.emit(SIGNAL(self.signal))#, convImg(imgFace,120,120), convImg(db_im,120,120), minkey)
            else:
                pass
                #self.emit(SIGNAL(self.signal), convImg(cv2.imread('nopic.jpg'), 120, 120),'../Database/Stranger.jpg', "Stranger")
                #time.sleep(0.04)
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class IdcardTimer(QThread):
    
    def __init__(self, qFace = Queue.Queue(4),
    			signal = "updateTime", parent=None):
        super(IdcardTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            time.sleep(0.05)
            if self.stoped:
                return
            begin = time.clock()
            id_num, byte_im, picklePath, name = readIDcard()
            if id_num != 0:
                if not qFace.empty():
                    img = qFace.get(True,3)
                    #print ("type is {}".format(type(img)))
                    if isinstance(img, types.NoneType):
                        print "Capture is not a picture."
                        self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                        time.sleep(1)
                    else:
                        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                        dets = detector(gray)
                        if(len(dets)==0):
                            print "No face detected"
                            self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                            time.sleep(1)
                        else:
                            f = open(picklePath,'rb')
                            DB=pickle.load(f)
                            f.close()
                            flag = 1
                            for k, d in enumerate(dets):
                                [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                                if x1<0:
                                    x1 = 10
                                #print ("img_shape = {}".format(img.shape))
                                imgFace = img[y1:y2,x1:x2,:]
                                minkey, minscore = faceRec(img,gray,d,DB)
                                if minkey != "Stranger":
                                    s = str(int(minscore))+" "+minkey
                                    s = "Successful!"
                                    self.emit(SIGNAL(self.signal), s+' '+name, byte_im, convImg(imgFace,220,300))
                                    print s
                                    time.sleep(1)
                                    flag = 0
                                    break
                            if flag==1:
                                self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                                print "Stranger."
                                time.sleep(1)
                            end = time.clock()
                            print ("Whole id time is {}".format(end-begin))

            else:
                self.emit(SIGNAL(self.signal), "Please show your ID card.", 0, 0)
                print "Please show your ID card."
                time.sleep(1)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped


qFace = Queue.Queue(1)

qnum = 3
qDet = Queue.Queue(qnum)
qGray = Queue.Queue(qnum)
qImg = Queue.Queue(qnum)
qImgFace = Queue.Queue(qnum)

qFeatNum = 3
qFeat = Queue.Queue(qFeatNum)
qFeat_Img = Queue.Queue(qFeatNum)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow(qFace, qDet, qGray, qImg, qImgFace, qFeat, qFeat_Img)
    mainwindow.show()
    sys.exit(app.exec_())