#coding=utf-8
from import_by_IdcardWindowThread import *
caffe.set_mode_gpu()

#capture = cv2.VideoCapture('../Testdata/20170528.avi')
#capture = cv2.VideoCapture('../Testdata/2018_03_12_16_03_18.avi')
#capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"DIVX")#(*"XVID")
saveVideoPath = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi'
saveVideoPath2 = 'CaptureRecording1.avi'
out = cv2.VideoWriter(saveVideoPath,fourcc,10,(640,480))#(1920,1080)) # 10 is speed. 640,480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)


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
    def __init__(self, qFace):
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

        self.playtimer = Timer("videoPlay", self.qFace)
        self.idcardtimer = IdcardTimer(self.qFace, "idcardPlay")
        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)

        self.connect(self.idcardtimer, SIGNAL("idcardPlay"), self.playIdcard)
        self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        self.connect(self.playButton, SIGNAL("clicked()"), self.VideoPlayPause)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self.exit)
        self.setWindowTitle('Face Recognition')
        #self.resize(1170, 800)
        self.resize(1100, 830)

    def playVideo(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))

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
        while True:
            if self.stoped:
                return
            begin = time.clock()
            ret, face = capture.read()
            face = cv2.flip(face, 1)
            out.write(face)
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
                                    end = time.clock()
                                    print ("obtain result time is {}".format(end-begin))
                                    self.emit(SIGNAL(self.signal), s+' '+name, byte_im, convImg(imgFace,220,300))
                                    print s
                                    time.sleep(1)
                                    flag = 0
                                    break
                            if flag==1:
                                end = time.clock()
                                print ("obtain result time is {}".format(end-begin))
                                self.emit(SIGNAL(self.signal), "Failed.", byte_im, 0)
                                print "Stranger."
                                time.sleep(1)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow(qFace)
    mainwindow.show()
    sys.exit(app.exec_())