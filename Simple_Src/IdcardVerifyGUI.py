#coding=utf-8
from improted_by_IdcardVerifyGUI import *
caffe.set_mode_gpu()

#capture = cv2.VideoCapture('../Testdata/20170528.avi')
#capture = cv2.VideoCapture('../Testdata/2018_03_12_16_03_18.avi')
#capture = cv2.VideoCapture(0)

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
        #begin = time.clock()
        pic_path = image_path + image_name[13]
        id_num = pic_path.split('/')[2].split('.')[0].split('_')[1]
        name = pic_path.split('/')[2].split('.')[0].split('_')[0]
        name = name.decode("gb2312")
        pic_path_new = 'D:/IDcard_test_program/'+id_num+'.jpg'
        shutil.move(pic_path, pic_path_new)
        imgID = cv2.imread(pic_path_new)
        byte_im = convImg(imgID,220,300)

        #picklePath = '../Models/'+id_num+'_DB.pickle'
        #createDatabase(id_num,pic_path_new,picklePath)
        #end = time.clock()
        #print ("Create Database time is {}".format(end - begin))

        #os.remove(pic_path_new)
        shutil.move(pic_path_new,'D:/testImage/history/'+name+id_num+'.jpg')
    else:
        id_num = 0
        byte_im = 0
        #picklePath = 0
        name = 0
        imgID = 0
    return id_num, byte_im, name, imgID


class MainWindow(QWidget):
    def __init__(self, qFace, qResult, terminateAll):
        super(MainWindow, self).__init__()
        self.status = 0
        
        self.playButton = QPushButton("Play")
        self.playButton.setStyleSheet('font-size:30px; border-radius:10px;border:3px groove gray')#;
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.setStyleSheet('font-size:30px; border-radius:10px;border:3px groove gray')
        self.exitbtn = QPushButton('Exit')
        self.exitbtn.setStyleSheet('font-size:30px; border-radius:10px;border:3px groove gray')

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


        self.qFace = qFace
        self.qResult = qResult
        self.terminateAll = terminateAll

        self.playtimer = Timer("videoPlay", self.qFace)
        self.idcardtimer = IdcardTimer(self.qResult, "idcardPlay")
        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)

        self.connect(self.idcardtimer, SIGNAL("idcardPlay"), self.playIdcard)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        self.connect(self.exitbtn, SIGNAL("clicked()"), self.exitFun)
        self.connect(self.playButton, SIGNAL("clicked()"), self.VideoPlayPause)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self.exit)
        self.setWindowTitle('Face Recognition')
        #self.resize(1170, 800)
        self.resize(1100, 830)

    def exitFun(self):
        self.close()
        self.terminateAll.value = not self.terminateAll.value

    def playVideo(self,face):
        byte_im = convImg(face,900,630)
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))

    #def playIdcard(self,s,id_byte_im,cap_byte_im):
    def playIdcard(self,resultList):
        s,id_byte_im,cap_byte_im = resultList
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
    
    def __init__(self, signal = "updateTime", qFace = multiprocessing.Queue(3), parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qFace = qFace

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False

        #capture = cv2.VideoCapture(0)
        capture = cv2.VideoCapture('../Testdata/v3.mov')
        #capture = cv2.VideoCapture('../Testdata/2018_03_12_16_03_18.avi')


        while True:
            if self.stoped:
                return
            #begin = time.clock()
            ret, face = capture.read()
            #out.write(face)
            face = cv2.flip(face, 1)
            cv2.resize(face,(640,480))
            #cv2.rectangle(face,(5,5),(635,475),(255,255,255),2)
            if not self.qFace.full():
                self.qFace.put(face)
                print "put face completed!"
            #else:
            #    qFace.get(True,3)
            #    qFace.put(face)


            #left->right:0-640(215-425)
            #top->bottom:0-480(115-36
            cv2.rectangle(face,(215,115),(425,365),(255,255,255),2)
            #font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(face,"Verify Region",(200,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,144,30),1,4,False)
            #byte_im = convImg(face,900,630)
            self.emit(SIGNAL(self.signal),face)
            time.sleep(0.04)
            #end = time.clock()
            #print("capture time is {}".format(end - begin))
    
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class IdcardTimer(QThread):
    
    def __init__(self, qResult = multiprocessing.Queue(3),
    			signal = "updateTime", parent=None):
        super(IdcardTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qResult = qResult

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        #cnt = 0
        while True:
            if not self.qResult.empty():
                self.emit(SIGNAL(self.signal),self.qResult.get(True,3))
            '''
            time.sleep(0.05)
            if self.stoped:
                return
            begin = time.clock()
            id_num, byte_im, name, imgID = readIDcard()
            if id_num != 0:
                end2 = time.clock()
                print ("reading ID card time is {}".format(end2 - begin))
                if not self.qFace.empty():
                    imgCap = self.qFace.get(True,3)
                    score,faceCap = compareFace(imgCap,imgID,mean,compoT,A,G)
                    #if score == 0:
                    #    print "score is zero!"
                    #    self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                    #  time.sleep(1)
                    #    continue
                    if score < 140:
                        print "Failed! You are stranger."
                        cnt = 0
                        self.emit(SIGNAL(self.signal), "Failed."+str(int(score)), byte_im, 0)
                        end = time.clock()
                        print ("reading & calculating time is {}".format(end - begin))
                        #time.sleep(0.5)
                        continue
                    if score > 140:
                        print "Successful! "+ name + id_num
                        cnt = 0
                        self.emit(SIGNAL(self.signal), ' Successful! '+name+' '+id_num+str(int(score)), byte_im, convImg(faceCap,220,300))
                        end = time.clock()
                        print ("reading & calculating time is {}".format(end - begin))
                        #time.sleep(0.5)
            else:
                cnt += 1
                if cnt == 60:
                    cnt = 0
                    self.emit(SIGNAL(self.signal), "Please show your ID card.", 0, 0)
            '''

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped



class VerifyProcess(multiprocessing.Process):
    def __init__(self,qFace,qResult):
        multiprocessing.Process.__init__(self)
        self.qFace = qFace
        self.qResult = qResult
    def run(self):
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

        with open('../Jb/' + 'A.pkl', 'rb') as f:
            A = pickle.load(f)
        with open('../Jb/' + 'G.pkl', 'rb') as f:
            G = pickle.load(f)

        saveMeanPath = '../Jb/mean.mat'
        saveCompoTPath = '../Jb/compoT.mat'
        #savemat(saveMeanPath,{'mean':mean})
        #savemat(saveCompoTPath,{'compoT':compoT})
        mean = loadmat(saveMeanPath)['mean']
        compoT = loadmat(saveCompoTPath)['compoT']

        print "load net and constant values completed!"
        threshold = 80
        cnt = 0
        while True:
            time.sleep(0.5)
            begin = time.clock()
            id_num, byte_im, name, imgID = readIDcard()
            if id_num != 0:
                end2 = time.clock()
                print ("reading ID card time is {}".format(end2 - begin))
                if not self.qFace.empty():
                    imgCap = self.qFace.get(True,3)
                    score,faceCap = compareFace(imgCap,imgID,net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth,mean,compoT,A,G)
                    #if score == 0:
                    #    print "score is zero!"
                    #    self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                    #  time.sleep(1)
                    #    continue
                    if score < threshold:
                        print "Failed! You are stranger."
                        cnt = 0
                        #self.qResult.put(["Failed. "+str(int(score)), byte_im, convImg(faceCap,220,300)])
                        self.qResult.put(["Failed.", byte_im, convImg(faceCap,220,300)])
                        end = time.clock()
                        print ("reading & calculating time is {}".format(end - begin))
                        #time.sleep(0.5)
                        continue
                    if score > threshold:
                        print "Successful! "+ name + id_num
                        cnt = 0
                        #self.qResult.put([name+' '+id_num+' '+str(int(score)), byte_im, convImg(faceCap,220,300)])
                        self.qResult.put([name+' '+id_num, byte_im, convImg(faceCap,220,300)])
                        end = time.clock()
                        print ("reading & calculating time is {}".format(end - begin))
                        #time.sleep(0.5)
            else:
                cnt += 1
                if cnt == 6:
                    cnt = 0
                    print "clear!!!!!!!!!!!!"
                    self.qResult.put(["Please show your ID card.", 0, 0])
                    #self.emit(SIGNAL(self.signal), "Please show your ID card.", 0, 0)

class ShowWindow(multiprocessing.Process):
    def __init__(self,qFace,qResult,terminateAll):
        multiprocessing.Process.__init__(self)
        self.qFace = qFace
        self.terminateAll = terminateAll
        self.qResult = qResult
    def run(self):
        app = QApplication(sys.argv)
        mainwindow = MainWindow(self.qFace,self.qResult,self.terminateAll)
        mainwindow.show()
        sys.exit(app.exec_())

qFace = multiprocessing.Queue(1)
qResult = multiprocessing.Queue(1)
qRead = multiprocessing.Queue(1)

terminateAll = multiprocessing.Value("i", 0)

if __name__ == '__main__':
    showwindow = ShowWindow(qFace,qResult,terminateAll)
    verifyprocess = VerifyProcess(qFace,qResult)
    #compareprocess = CompareProcess(qFace,qResult,qRead)
    #readprocess = ReadProcess(qRead)
    showwindow.start()
    verifyprocess.start()
    #compareprocess.start()
    #readprocess.start()

    while True:
        if terminateAll.value:
            showwindow.terminate()
            verifyprocess.terminate()
            #compareprocess.terminate()
            #readprocess.terminate()
            del showwindow,verifyprocess#,compareprocess,readprocess
            break