#coding=utf-8
from imported_by_FaceRecGUI import *
from mxnetRec import *
caffe.set_mode_gpu()


#capture = cv2.VideoCapture('../Testdata/2018_03_12_16_03_18.avi')
#capture = cv2.VideoCapture('../Testdata/20170528.avi')

#capture = cv2.VideoCapture(0)

'''
fourcc = cv2.VideoWriter_fourcc(*"DIVX")#(*"XVID")
saveVideoPath = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi'
saveVideoPath2 = 'CaptureRecording1.avi'
out = cv2.VideoWriter(saveVideoPath,fourcc,20,(640,480))#(1920,1080)) # 10 is speed. 640,480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
'''

def convImg(img,w=100,h=100):
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
        try:
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
            os.remove(pic_path_new)
        except:
            id_num = 0
            byte_im = 0
            #picklePath = 0
            name = 0
            imgID = 0


            print "picture name error, removing......"
            for tmp in image_name:
                if tmp.split('.')[-1] == 'jpg':
                    os.remove(image_path+tmp)
                if tmp.split('.')[-1] == 'bmp':
                    os.remove(image_path+tmp)
            
            print "completede!"
            print "continue......"
    elif image_name.__len__()==13:
        id_num = 0
        byte_im = 0
        #picklePath = 0
        name = 0
        imgID = 0
        #print "Waiting for idcard picture......"
    else:
        id_num = 0
        byte_im = 0
        #picklePath = 0
        name = 0
        imgID = 0

        print "more than 14 files, removing......"
        for tmp in image_name:
            if tmp.split('.')[-1] == 'jpg':
                os.remove(image_path+tmp)
            if tmp.split('.')[-1] == 'bmp':
                os.remove(image_path+tmp)
        print "completede!"
        print "continue......"

    return id_num, byte_im, name, imgID



'''
def cropFace(img):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    dets = detector(gray)
    for k, d in enumerate(dets):
        [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
        imgFace = img[y1:y2,x1:x2,:]
        break
    return imgFace

def name2dbface(name):
    db_im = cv2.imread('../Database/'+name+'.jpg')
    db_im_face = cropFace(db_im)
    return convImg(db_im_face,100,100)
'''
def name2dbface(name):
    db_im_face = cv2.imread('../DatabaseFace/'+name+'.jpg')
    return convImg(db_im_face,100,100)

class MainWindow(QWidget):
    def __init__(self, qFace, qList, qPost, qCapture, qSmallFace, qResult, qSJTU1, qSJTU2, terminateAll):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Face Recognition')
        self.setWindowIcon(QIcon('sjtu.png'))
        self.resize(1200, 850)
        self.status = 0
        self.DBstatus = 0

        self.playButton = QPushButton("Play")
        self.playButton.setStyleSheet('font-size:20px; border-radius:10px;border:3px groove gray')
        self.cancelButton = QPushButton("UpdateDB")
        self.cancelButton.setStyleSheet('font-size:20px; border-radius:10px;border:3px groove gray')
        self.exitbtn = QPushButton('Exit')
        self.exitbtn.setStyleSheet('font-size:20px; border-radius:10px;border:3px groove gray')
        self.exitbtn.setFixedWidth(300)  
        self.exitbtn.setFixedHeight(30)
        self.playButton.setFixedWidth(300)  
        self.playButton.setFixedHeight(30)
        self.cancelButton.setFixedWidth(300)  
        self.cancelButton.setFixedHeight(30)

        self.image = QImage()
        self.imageLabel = QLabel('')

        ######## multi_face_show ############
        self.face11 = QLabel('')
        self.face12 = QLabel('')
        self.text1 = QLabel(" ")
        self.setTextFont(self.text1, w = 250, h = 30, s = 15)
        self.face21 = QLabel('')
        self.face22 = QLabel('')
        self.text2 = QLabel(" ")
        self.setTextFont(self.text2, w = 250, h = 30, s = 15)
        self.face31 = QLabel('')
        self.face32 = QLabel('')
        self.text3 = QLabel(" ")
        self.setTextFont(self.text3, w = 250, h = 30, s = 15)
        self.face41 = QLabel('')
        self.face42 = QLabel('')
        self.text4 = QLabel(" ")
        self.setTextFont(self.text4, w = 250, h = 30, s = 15)
        self.face51 = QLabel('')
        self.face52 = QLabel('')
        self.text5 = QLabel(" ")
        self.setTextFont(self.text5, w = 250, h = 30, s = 15)

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
        self.f5 = QHBoxLayout()
        self.f5.addWidget(self.face51)
        self.f5.addWidget(self.face52)
        self.side = QVBoxLayout()
        self.side.addLayout(self.f1)
        self.side.addWidget(self.text1)
        self.side.addLayout(self.f2)
        self.side.addWidget(self.text2)
        self.side.addLayout(self.f3)
        self.side.addWidget(self.text3)
        self.side.addLayout(self.f4)
        self.side.addWidget(self.text4)
        self.side.addLayout(self.f5)
        self.side.addWidget(self.text5)
        ######## multi_face_show ############


        ######## idcard images begin ##########
        self.idcardim1 = QLabel('')
        self.idcardim2 = QLabel('')
        self.idtext = QLabel(" ")
        self.setTextFont(self.idtext, w = 220, h = 80, s = 25)

        self.idcard = QVBoxLayout()
        self.idcard.addWidget(self.idcardim1)
        self.idcard.addWidget(self.idcardim2)
        self.idcard.addWidget(self.idtext)

        ######## idcard images begin ##########

        ######## bottom images begin ##############
        self.c1 = QLabel('')
        self.c2 = QLabel('')
        self.c3 = QLabel('')
        self.c4 = QLabel('')
        self.c5 = QLabel('')
        self.c6 = QLabel('')
        self.c7 = QLabel('')
        self.c8 = QLabel('')
        self.c9 = QLabel('')
        self.c10 = QLabel('')
        self.c11 = QLabel('')
        self.c12 = QLabel('')

        self.bc = QHBoxLayout() #bottom capture image
        self.bc.addWidget(self.c1)
        self.bc.addWidget(self.c2)
        self.bc.addWidget(self.c3)
        self.bc.addWidget(self.c4)
        self.bc.addWidget(self.c5)
        self.bc.addWidget(self.c6)
        self.bc.addWidget(self.c7)
        self.bc.addWidget(self.c8)
        self.bc.addWidget(self.c9)
        self.bc.addWidget(self.c10)
        self.bc.addWidget(self.c11)
        self.bc.addWidget(self.c12)

        self.b_cnt = 1 #bottom image count
        self.faces = []
        ######## bottom images end ##############


        self.videoImage = QHBoxLayout()
        self.videoImage.addWidget(self.imageLabel)
        self.videoImage.addLayout(self.side)
        self.videoImage.addLayout(self.idcard)
        
        self.buttons = QHBoxLayout()
        #hbox.addStretch(1)
        self.buttons.addWidget(self.playButton)
        self.buttons.addWidget(self.cancelButton)
        self.buttons.addWidget(self.exitbtn)

        self.whole = QVBoxLayout()
        self.whole.addLayout(self.videoImage)
        self.whole.addLayout(self.bc)
        self.whole.addLayout(self.buttons)
        
        self.setLayout(self.whole)

        self.qFace = qFace
        self.qList = qList
        self.qPost = qPost
        self.qCapture = qCapture
        self.qSmallFace = qSmallFace
        self.qResult = qResult
        # self.featureDict = featureDict
        self.qSJTU1 = qSJTU1
        self.qSJTU2 = qSJTU2
        self.terminateAll = terminateAll

        self.playtimer = Timer("videoPlay", self.qCapture)
        self.facetimer = FaceTimer("facePlay", self.qSmallFace)
        self.texttimer = TextTimer("textPlay", self.qPost)
        self.idcardtimer = IdcardTimer("idcardPlay",self.qResult)
        self.databasetimer = DatabaseTimer("databaseUpdate",self.qCapture,qSJTU1,qSJTU2)

        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)
        self.connect(self.facetimer, SIGNAL("facePlay"), self.playFace)
        self.connect(self.texttimer, SIGNAL("textPlay"), self.multiFaceNew)
        self.connect(self.idcardtimer, SIGNAL("idcardPlay"), self.playIdcard)
        self.connect(self.databasetimer, SIGNAL("databaseUpdate"), self.updateDatabase)
        
        self.connect(self.playButton, SIGNAL("clicked()"), self.VideoPlayPause)
        self.connect(self.cancelButton, SIGNAL("clicked()"), self.DatabaseUpdataStop)
        self.connect(self.exitbtn, SIGNAL("clicked()"), self.exitFun)
        #self.connect(self.exitbtn, SIGNAL("clicked()"), self,SLOT('close()'))
        
        self.arr = [["Stranger",0,0],["Stranger",0,0],["Stranger",0,0],["Stranger",0,0],["Stranger",0,0]]
        '''
        self.playtimer.start()
        self.facetimer.start()
        self.texttimer.start()
        '''

    #three QThreads of Three blocks of GUI 
    def playIdcard(self,resultList):
        s,id_byte_im,cap_byte_im = resultList
    	self.idtext.setText(s)
        #self.textLabel2.setText(QString(s+u"中文"))
        if id_byte_im !=0:
            self.image.loadFromData(QByteArray(id_byte_im))
            self.idcardim2.setPixmap(QPixmap.fromImage(self.image))
        else:
            self.idcardim2.setText(" ")
        if cap_byte_im !=0:
            self.image.loadFromData(QByteArray(cap_byte_im))
            self.idcardim1.setPixmap(QPixmap.fromImage(self.image))
        else:
            self.idcardim1.setText(" ")
    # def playIdcard(self):
    #     self.idtext.setText("Peter Hou")
    #     id_im = cv2.imread("1.bmp")
    #     id_byte_im = convImg(id_im,220,300)
    #     self.image.loadFromData(QByteArray(id_byte_im))
    #     self.idcardim1.setPixmap(QPixmap.fromImage(self.image))
    #     self.image.loadFromData(QByteArray(id_byte_im))
    #     self.idcardim2.setPixmap(QPixmap.fromImage(self.image))

    def playVideo(self,face):
        byte_im = convImg(face,1000,700)
        self.image.loadFromData(QByteArray(byte_im))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
    def playFace(self,byte_im):
        self.image.loadFromData(QByteArray(byte_im))
        if self.b_cnt == 1:
            self.c1.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 2:
            self.c2.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 3:
            self.c3.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 4:
            self.c4.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 5:
            self.c5.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 6:
            self.c6.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 7:
            self.c7.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 8:
            self.c8.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 9:
            self.c9.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 10:
            self.c10.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 11:
            self.c11.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 12:
            self.c12.setPixmap(QPixmap.fromImage(self.image))
            self.b_cnt += 1
            self.faces += [byte_im]
        if self.b_cnt == 13:
            self.image.loadFromData(QByteArray(self.faces[1]))
            self.c1.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[2]))
            self.c2.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[3]))
            self.c3.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[4]))
            self.c4.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[5]))
            self.c5.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[6]))
            self.c6.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[7]))
            self.c7.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[8]))
            self.c8.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[9]))
            self.c9.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[10]))
            self.c10.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(self.faces[11]))
            self.c11.setPixmap(QPixmap.fromImage(self.image))
            self.image.loadFromData(QByteArray(byte_im))
            self.c12.setPixmap(QPixmap.fromImage(self.image))
            for i in range(0,11):
                self.faces[i] = self.faces[i+1]
            self.faces[11] = byte_im
    def multiFaceNew(self, aGet):
        if aGet[0][0:8] == "Stranger":
            aGet[0] = "Stranger"
        if aGet[0] != "Stranger":
        #if aGet[3] < 1.09:
            self.checkExist(aGet, self.arr)
        self.checkDisappear(self.arr)
        self.showSide(self.arr)
    
    #one database update Qthread
    def updateDatabase(self):
        self.DBstatus = 0
        self.cancelButton.setText("UpdateDB")
        self.databasetimer.stop()
        print "Updating database completed!"
    
    #three buttons
    '''
    def VideoPlayPause(self):
        self.status, playstr = ((1, 'Play'), (0, 'Pause'))[self.status]
        self.playButton.setText(playstr)
        if self.status is 1:
            self.playtimer.stop()
            self.facetimer.stop()
            self.texttimer.stop()
        else:
            self.playtimer.start()
            self.facetimer.start()
            self.texttimer.start()
    '''
    def VideoPlayPause(self):
        self.status, playstr = ((1, 'Pause'), (0, 'Play'))[self.status]
        self.playButton.setText(playstr)
        if self.status is 1:
            self.playtimer.start()
            self.facetimer.start()
            self.texttimer.start()
            self.idcardtimer.start()

        else:
            self.playtimer.stop()
            self.facetimer.stop()
            self.texttimer.stop()
            self.idcardtimer.stop()


    def DatabaseUpdataStop(self):
        self.DBstatus, cancelstr = ((1, 'Updating...'), (0, 'UpdateDB'))[self.DBstatus]
        self.cancelButton.setText(cancelstr)
        if self.DBstatus is 1:
            self.databasetimer.start()
        else:
            self.databasetimer.stop()
    def exitFun(self):
        self.close()
        self.terminateAll.value = not self.terminateAll.value

    #supporting functions
    def showSide(self,a):
        for i in range(0,5):
            if a[i][0] == "Stranger":
                self.clearLabel(i+1)
            else:
                self.whichToShow(i+1, a[i][2], name2dbface(a[i][0]), a[i][0])
    def checkDisappear(self,a):
        for i in range(0,5):
            if a[i][0] == "Stranger":
                break
            else:
                a[i][1] +=1
        for i in range(0,5):
            if a[i][1] == 5:
                self.shift(i,a)
                i=i-1
    def checkExist(self,aGet,a):
        for i in range(0,5):
            if a[i][0] == "Stranger":
                a[i] = aGet
                break
            if aGet[0] == a[i][0]:
                a[i] = aGet
                break
            if i == 4:
                a[0] = aGet
    def shift(self,num,a):
        if num ==4:
            a[4][0] = "Stranger"
            a[4][1] = 0
        else:
            for i in range(num,4):
                a[i] = a[i+1]
            a[4][0] = "Stranger"
            a[4][1] = 0
    def showResult(self, imLabel1, imLabel2, tLabel, im, db, name):
        self.image.loadFromData(QByteArray(im))
        imLabel1.setPixmap(QPixmap.fromImage(self.image))
        self.image.loadFromData(QByteArray(db))
        imLabel2.setPixmap(QPixmap.fromImage(self.image))
        tLabel.setText(name)
    def whichToShow(self,num, im, db, name):
        if num == 1:
            self.showResult(self.face11, self.face12, self.text1, im, db, name)
        elif num ==2:
            self.showResult(self.face21, self.face22, self.text2, im, db, name)
        elif num == 3:
            self.showResult(self.face31, self.face32, self.text3, im, db, name)
        elif num == 4:
            self.showResult(self.face41, self.face42, self.text4, im, db, name)
        elif num == 5:
            self.showResult(self.face51, self.face52, self.text5, im, db, name)
    def clearLabel(self,num):
        if num == 1:
            self.face11.setText(" ")
            self.face12.setText(" ")
            self.text1.setText(" ")
        elif num ==2:
            self.face21.setText(" ")
            self.face22.setText(" ")
            self.text2.setText(" ")
        elif num == 3:
            self.face31.setText(" ")
            self.face32.setText(" ")
            self.text3.setText(" ")
        elif num == 4:
            self.face41.setText(" ")
            self.face42.setText(" ")
            self.text4.setText(" ")
        elif num == 5:
            self.face51.setText(" ")
            self.face52.setText(" ")
            self.text5.setText(" ")
    def setTextFont(self,l, w = 250, h = 30, s = 8):
    	l.setFixedWidth(w)  
        l.setFixedHeight(h)  
        l.setAlignment(Qt.AlignCenter)
        l.setFont(QFont("Arial",s,QFont.Normal))
        #l.setFont(QFont("Roman times",s,QFont.Bold))
    def setDefaultPic(self,l,w = 100,h = 100,im = 'nopic.jpg'):
    	orig_im = cv2.imread(im)
        byte_im = convImg(orig_im,w,h)
        self.image.loadFromData(QByteArray(byte_im))
        l.setPixmap(QPixmap.fromImage(self.image))
        #l.setScaledContents(True)

class Timer(QThread):
    def __init__(self, signal = "updateTime", qCapture = multiprocessing.Queue(3), parent=None):
        super(Timer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qCapture = qCapture


    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False     
        while True:
            time.sleep(0.04)
            #time.sleep(0.2)
            if self.stoped:
                return
            if not self.qCapture.empty():
                face = self.qCapture.get(True,3)
                self.emit(SIGNAL(self.signal),face)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(self.mutex):
            return self.stoped

class FaceTimer(QThread):
    def __init__(self, signal = "updateTime", qSmallFace = multiprocessing.Queue(3), parent=None):
        super(FaceTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qSmallFace = qSmallFace
    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            time.sleep(0.1)
            if self.stoped:
                return
            if not self.qSmallFace.empty():
                self.emit(SIGNAL(self.signal), self.qSmallFace.get(True,3))

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(self.mutex):
            return self.stoped

class TextTimer(QThread):
    def __init__(self, signal = "updateTime", qPost = multiprocessing.Queue(3), parent=None):
        super(TextTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qPost = qPost

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            time.sleep(0.1)
            if self.stoped:
                return
            if not self.qPost.empty():
                self.emit(SIGNAL(self.signal), self.qPost.get(True,3))

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(self.mutex):
            return self.stoped

class IdcardTimer(QThread):
    
    def __init__(self, signal = "updateTime", qResult = multiprocessing.Queue(3), parent=None):
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

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped


class DatabaseTimer(QThread):
    def __init__(self, signal = "updateTime", qCapture = multiprocessing.Queue(3), qSJTU1 = multiprocessing.Queue(100), qSJTU2 = multiprocessing.Queue(100), parent=None):
        super(DatabaseTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()
        self.qCapture = qCapture
        # self.featureDict = featureDict
        self.qSJTU1 = qSJTU1
        self.qSJTU2 = qSJTU2
        self.sjtuCnt = 1
        self.args = get_parser()
        self.face_identify = FaceIdendify(self.args)

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        print "run..............."

        while True:
            if self.stoped:
                return
            if not self.qCapture.empty():
                face = self.qCapture.get(True,3)
                cv2.imwrite('../DatabaseSJTU/'+'SJTU'+str(self.sjtuCnt)+'.jpg',face)
                cv2.imshow('SJTU'+str(self.sjtuCnt),face)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()

                #img = cv2.imread('../DatabaseSJTU/'+'SJTU'+str(self.sjtuCnt)+'.jpg')
                feats, dets, points_all = self.face_identify.get_features(face)
                try:
                    d = dets[0]
                    [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                    imgFace = face[y1:y2,x1:x2,:]
                    cv2.imwrite('../DatabaseFace/'+'SJTU'+str(self.sjtuCnt)+'.jpg',imgFace)
                    for i, feat in enumerate(feats):
                        #self.featureDict['SJTU'+str(self.sjtuCnt)] = feat
                        self.qSJTU1.put('SJTU'+str(self.sjtuCnt))
                        self.qSJTU2.put(feat)
                        self.sjtuCnt+=1
                        break
                except:
                    print "please stand in front of the camera!!!"
                # for i in range(int(self.qSJTU1.qsize())):
                #     print self.qSJTU1[i]
                #     print self.qSJTU2[i]
                #     #print value
                

                # args = get_parser()
                # face_identify = FaceIdendify(args)
                # face_identify.add_student(self.stuCnt,face)

               
                self.emit(SIGNAL(self.signal))
                break

        '''
        while True:
            if self.stoped:
                return
            begin = time.clock()
            args = get_parser()
            face_identify = FaceIdendify(args)
            end = time.clock()
            print("loading MXNet time is {}".format(end - begin))

            print "Updating Database......"

            face_identify.create_database()

            #databaseUpdate()
            self.emit(SIGNAL(self.signal))
            break
        '''

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class CaptureProcess(multiprocessing.Process):
    def __init__(self,qCapture,qFace,qIdface):
        multiprocessing.Process.__init__(self)
        self.qCapture = qCapture
        self.qFace = qFace
        self.qIdface = qIdface
    def run(self):
        #capture = cv2.VideoCapture('../Testdata/20170528.avi')
        #capture = cv2.VideoCapture('../Testdata/v3.mov')
        #capture = cv2.VideoCapture('../starsVideo/v1.mov')
        capture = cv2.VideoCapture(0)
        #capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        #capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #out = cv2.VideoWriter(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi',cv2.VideoWriter_fourcc(*"DIVX"),20,(640,480))
        # cnt = 1
        while True:
            #begin = time.clock()
            #time.sleep(0.005)
            #time.sleep(0.05)
            #for i in range(1):
            ret, face = capture.read()

            if ret == True:
                face = cv2.flip(face, 1)
                #out.write(face)
                #byte_im = convImg(face,1000,700)
                #print ("capture size is {}".format(face.shape))
                # cv2.imwrite('shots/'+str(cnt)+'.jpg',face)
                # cnt += 1

                #it not necessary to judge whether the queue is full,
                #because when the FIFO-queue is full, 
                #it will remove the head item and put the new itme at the tail

                # if not self.qCapture.full():
                #     self.qCapture.put(face)

                # if not self.qFace.full():
                #     self.qFace.put(face)

                self.qCapture.put(face)
                self.qFace.put(face)
                if not self.qIdface.full():
                    self.qIdface.put(face)
                #self.qIdface.put(face)
                
                #end = time.clock()
                #print("capture time is {}".format(end - begin))

class SmallFace(multiprocessing.Process):
    def __init__(self,qFace,qList,qSmallFace):
        multiprocessing.Process.__init__(self)
        self.qFace = qFace
        self.qList = qList
        self.qSmallFace = qSmallFace
    def run(self):
        while True:
            if not self.qFace.empty():
                img = self.qFace.get(True,3)
                if isinstance(img, types.NoneType):
                    continue
                else:
                    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                    dets = detector(gray)

                    #list1 = [img,gray,dets]
                    list1 = img
                    if not self.qList.full():
                            self.qList.put(list1)

                    if(len(dets)==0):
                        continue
                    else:
                        for k, d in enumerate(dets):
                            [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                            if x1<0:
                                x1 = 10
                            if y1<0:
                                y1 = 10
                            imgFace = img[y1:y2,x1:x2,:]
                            if not self.qSmallFace.full():
                                self.qSmallFace.put(convImg(imgFace,100,100))

class RecProcess(multiprocessing.Process):
    def __init__(self, qList, qPost, qSJTU1, qSJTU2, loadNetEndFlag):
        multiprocessing.Process.__init__(self)
        self.qList = qList
        self.qPost = qPost
        self.qSJTU1 = qSJTU1
        self.qSJTU2 = qSJTU2
        self.loadNetEndFlag = loadNetEndFlag

        self.featureDict = {'Name': 'Feature'};
        self.featureDict.clear();

    def run(self):
        begin = time.clock()
        args = get_parser()
        face_identify = FaceIdendify(args)
        end = time.clock()

        print("loading MXNet(multiface) time is {}".format(end - begin))

        print "you can now click play to start(multiface)!"

        self.loadNetEndFlag.value = not self.loadNetEndFlag.value



        #fileRes = r'D:/face_recognition/Simple_Src/printName1.txt'
        #cnt = 1
        '''
        while True:
            if not self.qList.empty():
                img, gray, dets = self.qList.get(True,3)
                if(len(dets)==0):
                    print "No face detected"
                    aPost = ["Stranger",0,0]
                    if not self.qPost.full():
                        self.qPost.put(aPost)
                    continue
                for k, d in enumerate(dets):
                    [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
                    if x1<0:
                        x1 = 10
                    if y1<0:
                        y1 = 10
                    imgFace = img[y1:y2,x1:x2,:]
                    minkey, minscore = faceRec(net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth,img,gray,d)
                    aPost = [minkey,0,convImg(imgFace,100,100)]

                    #with open(fileRes,'a+') as fr:
                    #    fr.write("No. {} name is {}".format(cnt,minkey)+'\r')
                    #cnt += 1
                    if not self.qPost.full():
                        self.qPost.put(aPost)
        '''
        while True:
            if not self.qList.empty():
                img = self.qList.get(True,3)
                begin = time.clock()
                #print len(self.qSJTU1)

                #print self.qSJTU1.qsize()
                if not self.qSJTU1.empty():
                    self.featureDict[self.qSJTU1.get()] = self.qSJTU2.get()
                print len(self.featureDict)
                print self.featureDict.keys()
                # resultList = mxnetFaceRec(img,face_identify) # minkey,minscore,imgFace
                if len(self.featureDict) > 0:
                    resultList = mxnetFaceRec_new(img,face_identify,self.featureDict)
                    #print "newwwwwwwwwwwwwwwww!"
                else:
                    resultList = mxnetFaceRec(img,face_identify) # minkey,minscore,imgFace
                    #print "olddddddddddddddddd!"
                end = time.clock()
                print ("mxnet calculate time(from img to minkey) is {}".format(end-begin))
                #aPost = [minkey,0,convImg(imgFace,100,100)]
                for i in range(len(resultList)):
                    aPost = [resultList[i][0],0,convImg(resultList[i][2],100,100)]
                    if not self.qPost.full():
                        self.qPost.put(aPost)


class VerifyProcess(multiprocessing.Process):
    def __init__(self,qIdface,qResult,loadNetEndFlag2):
        multiprocessing.Process.__init__(self)
        self.qIdface = qIdface
        self.qResult = qResult
        self.loadNetEndFlag2 = loadNetEndFlag2
    def run(self):
        begin = time.clock()
        args = get_parser()
        face_identify = FaceIdendify(args)
        end = time.clock()

        print("loading MXNet(idcard) time is {}".format(end - begin))
        print "you can now click play to start(idcard)!"
        self.loadNetEndFlag2.value = not self.loadNetEndFlag2.value

        threshold = 1.09
        cnt = 0
        while True:
            time.sleep(0.5)
            begin = time.clock()
            id_num, byte_im, name, imgID = readIDcard()
            if id_num != 0:
                end2 = time.clock()
                print ("reading ID card time is {}".format(end2 - begin))
                s = [100,100,100]
                f = [0,0,0]
                flag_true = 0
                flag_false = 0
                for i in range(3):
                    if not self.qIdface.empty():
                        imgCap = self.qIdface.get(True,3)
                        bb = time.clock()
                        s[i],f[i] = mxnetCompare(imgCap,imgID,face_identify)
                        ee = time.clock()
                        print ("one face time is {}".format(ee - bb))
                        print ("the {}-th score is {}".format(i+1,s[i]))
                        if s[i]<threshold:
                            flag_true+=1
                        else:
                            flag_false+=1
                        if flag_true == 2:
                            score = s[i]
                            faceCap = f[i]
                            break
                        if flag_false == 2:
                            score = s[i]
                            faceCap = f[i]
                            break
                if score > threshold:
                    print "Failed! You are stranger."
                    cnt = 0
                    
                    #self.qResult.put(["Failed. "+str(round(score,2)), byte_im, convImg(faceCap,220,300)])
                    self.qResult.put(["Failed.", byte_im, convImg(faceCap,220,300)]) 
                    
                    end = time.clock()
                    print ("reading & calculating time is {}".format(end - begin))
                    #time.sleep(0.5)
                    continue
                if score < threshold:
                    print "Successful! "+ name + id_num
                    cnt = 0
                    self.qResult.put([name+' '+id_num+' '+str(round(score,2)), byte_im, convImg(faceCap,220,300)])
                    self.qResult.put([name, byte_im, convImg(faceCap,220,300)])
                    end = time.clock()
                    print ("reading & calculating time is {}".format(end - begin))
                    #time.sleep(0.5)
            else:
                if not self.qIdface.empty():
                    imgCap = self.qIdface.get(True,3)
                cnt += 1
                if cnt == 6:
                    cnt = 0
                    print "clear!!!!!!!!!!!!"
                    #self.qResult.put(["Waiting for ID card...", 0, 0])
                    self.qResult.put(["ID card...", 0, 0])
                    #self.emit(SIGNAL(self.signal), "Please show your ID card.", 0, 0)

class ShowWindow(multiprocessing.Process):
    def __init__(self,qFace,qList,qPost,qCapture,qSmallFace,qResult,qSJTU1,qSJTU2,terminateAll):
        multiprocessing.Process.__init__(self)
        self.qFace = qFace
        self.qList = qList
        self.qPost = qPost
        self.qCapture = qCapture
        self.terminateAll = terminateAll
        self.qSmallFace = qSmallFace
        self.qResult = qResult
        # self.featureDict = featureDict
        self.qSJTU1 = qSJTU1
        self.qSJTU2 = qSJTU2
    def run(self):
        app = QApplication(sys.argv)
        mainwindow = MainWindow(self.qFace, self.qList,self.qPost,self.qCapture,self.qSmallFace,self.qResult,self.qSJTU1,self.qSJTU2,self.terminateAll)
        mainwindow.show()
        sys.exit(app.exec_())

qFace = multiprocessing.Queue(2)
qList = multiprocessing.Queue(1)
qPost = multiprocessing.Queue(8)
qCapture = multiprocessing.Queue(10)
qSmallFace = multiprocessing.Queue(10)
qIdface = multiprocessing.Queue(3)
qResult = multiprocessing.Queue(1)

qSJTU1 = multiprocessing.Queue(5)# 1 is enough
qSJTU2 = multiprocessing.Queue(5)

print qSJTU1.qsize()


terminateAll = multiprocessing.Value("i", 0)
loadNetEndFlag = multiprocessing.Value("i", 0)
loadNetEndFlag2 = multiprocessing.Value("i", 0)

'''
fourcc = cv2.VideoWriter_fourcc(*"DIVX")#(*"XVID")
saveVideoPath = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi'
saveVideoPath2 = 'CaptureRecording1.avi'
out = cv2.VideoWriter(saveVideoPath,fourcc,10,(640,480))#(1920,1080)) # 10 is speed. 640,480
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
'''

if __name__ == '__main__':
    # featureDict=multiprocessing.Manager().dict()
    # featureDict = {'Name': 'Feature'};
    # featureDict.clear();
    # featureDict["gg"] = 111

    showwindow = ShowWindow(qFace,qList,qPost,qCapture,qSmallFace,qResult,qSJTU1,qSJTU2,terminateAll)
    recprocess = RecProcess(qList,qPost,qSJTU1,qSJTU2,loadNetEndFlag)
    captureprocess = CaptureProcess(qCapture,qFace,qIdface)
    smallface = SmallFace(qFace,qList,qSmallFace)
    verifyprocess = VerifyProcess(qIdface,qResult,loadNetEndFlag2)


    recprocess.start()
    verifyprocess.start()
    while True:
        if loadNetEndFlag.value and loadNetEndFlag2.value:
            captureprocess.start()
            smallface.start()
            showwindow.start()
            print "All processes have started."
            break
    while True:
        if terminateAll.value:
            captureprocess.terminate()
            recprocess.terminate()
            verifyprocess.terminate()
            smallface.terminate()
            showwindow.terminate()
            del showwindow,recprocess,verifyprocess,captureprocess,smallface
            break

'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow(qFace, qList,qPost,qCapture)#,playtimerOut,facetimerOut,texttimerOut,databasetimerOut)
    mainwindow.show()
    sys.exit(app.exec_())
'''
