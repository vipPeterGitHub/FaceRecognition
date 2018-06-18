#coding=utf-8
from imported_by_FaceRecGUI import *
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
        pic_path = image_path + image_name[13]
        id_num = pic_path.split('/')[2].split('.')[0].split('_')[1]
        pic_path_new = 'D:/IDcard_test_program/'+id_num+'.jpg'
        shutil.move(pic_path, pic_path_new)
        img = cv2.imread(pic_path_new)
        byte_im = convImg(img,220,300)
        picklePath = '../Models/'+id_num+'_DB.pickle'
        begin = time.clock()
        createDatabase(id_num,pic_path_new,picklePath)
        end = time.clock()
        print ("Create Database time is {}".format(end - begin))
        os.remove(pic_path_new)
    else:
        id_num = 0
        byte_im = 0
        picklePath = 0
    return id_num, byte_im, picklePath

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
    def __init__(self, qFace, qList, qPost, qCapture, qSmallFace, terminateAll):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Face Recognition')
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
        self.terminateAll = terminateAll

        self.playtimer = Timer("videoPlay", self.qCapture)
        self.facetimer = FaceTimer("facePlay", self.qSmallFace)
        self.texttimer = TextTimer("textPlay", self.qPost)
        self.databasetimer = DatabaseTimer("databaseUpdate")

        self.connect(self.playtimer, SIGNAL("videoPlay"), self.playVideo)
        self.connect(self.facetimer, SIGNAL("facePlay"), self.playFace)
        self.connect(self.texttimer, SIGNAL("textPlay"), self.multiFaceNew)
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
        if aGet[0] != "Stranger":
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

        else:
            self.playtimer.stop()
            self.facetimer.stop()
            self.texttimer.stop()

    def DatabaseUpdataStop(self):
        self.DBstatus, cancelstr = ((1, 'updating, please wait ...'), (0, 'UpdateDB'))[self.DBstatus]
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

class DatabaseTimer(QThread):
    def __init__(self, signal = "updateTime", parent=None):
        super(DatabaseTimer, self).__init__(parent)
        self.stoped = False
        self.signal = signal
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stoped = False
        while True:
            if self.stoped:
                return
            databaseUpdate()
            self.emit(SIGNAL(self.signal))
            break

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):    
        with QMutexLocker(sellf.mutex):
            return self.stoped

class CaptureProcess(multiprocessing.Process):
    def __init__(self,qCapture,qFace):
        multiprocessing.Process.__init__(self)
        self.qCapture = qCapture
        self.qFace = qFace
    def run(self):
        capture = cv2.VideoCapture('../Testdata/v3.mov')
        #capture = cv2.VideoCapture('../Testdata/e4.avi')
        #capture = cv2.VideoCapture(0)
        #capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        #capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        #out = cv2.VideoWriter(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) +'.avi',cv2.VideoWriter_fourcc(*"DIVX"),20,(640,480))
        while True:
            #begin = time.clock()
            ret, face = capture.read()
            if ret == True:
                face = cv2.flip(face, 1)
                #out.write(face)
                #byte_im = convImg(face,1000,700)

                #it not necessary to judge whether the queue is full,
                #because when the FIFO-queue is full, 
                #it will remove the head item and put the new itme at the tail
                self.qCapture.put(face)
                self.qFace.put(face)
                
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

                    list1 = [img,gray,dets]
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
    def __init__(self, qList, qPost, loadNetEndFlag):
        multiprocessing.Process.__init__(self)
        self.qList = qList
        self.qPost = qPost
        self.loadNetEndFlag = loadNetEndFlag
    def run(self):

        #pickleModlePath = '../Models/20180424_database600_caffe_hou.pickle'
        pickleModlePath = '../Models/20180426_db600_20stars_beijing&GOT_caffe_hou.pickle'
        #databasaPath = "../Database1600/600_model" 
        databasaPath = "../testFile656/Database656"

        if not os.path.exists(pickleModlePath):
            print "Databaes creating ......  Please wait......"
            databaseUpdate(databasaPath,pickleModlePath)
            print "Database creating completed!"
        else:
            print "Database pickle model has exists, loading ......"


        #import database
        #f = open('../Models/db_20170425_lab_qf.pickle','rb')
        #f = open('../Models/20180309_database_hou.pickle','rb')
        f = open(pickleModlePath,'rb')
        #f = open('../Models/20180309_database_hou_simple.pickle','rb')
        #f = open('../Models/610324199510253457_DB.pickle','rb')
        db_qf=pickle.load(f)
        f.close()
        print "Loading database pickle model completed!"




        caffe.set_mode_gpu()
        #caffe.set_device(0)

        begin = time.clock()
        #global net_ctf,net_downmouth,net_eye,net_le,net_re,net_wholeface,net_mouth
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




        end = time.clock()
        print("loading net time is {}".format(end - begin))

        print "you can now click play to start!"

        self.loadNetEndFlag.value = not self.loadNetEndFlag.value

        fileRes = r'D:/face_recognition/Simple_Src/printName1.txt'
        cnt = 1

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
                    minkey, minscore = faceRec(net_wholeface,net_ctf,net_le,net_re,net_eye,net_mouth,net_downmouth,img,gray,d,db_qf)
                    aPost = [minkey,0,convImg(imgFace,100,100)]

                    with open(fileRes,'a+') as fr:
                        fr.write("No. {} name is {}".format(cnt,minkey)+'\r')
                    cnt += 1
                    if not self.qPost.full():
                        self.qPost.put(aPost)

class ShowWindow(multiprocessing.Process):
    def __init__(self,qFace,qList,qPost,qCapture,qSmallFace,terminateAll):
        multiprocessing.Process.__init__(self)
        self.qFace = qFace
        self.qList = qList
        self.qPost = qPost
        self.qCapture = qCapture
        self.terminateAll = terminateAll
        self.qSmallFace = qSmallFace
    def run(self):
        app = QApplication(sys.argv)
        mainwindow = MainWindow(self.qFace, self.qList,self.qPost,self.qCapture,self.qSmallFace,self.terminateAll)
        mainwindow.show()
        sys.exit(app.exec_())

qFace = multiprocessing.Queue(3)
qList = multiprocessing.Queue(5)
qPost = multiprocessing.Queue(5)
qCapture = multiprocessing.Queue(10)
qSmallFace = multiprocessing.Queue(12)

terminateAll = multiprocessing.Value("i", 0)
loadNetEndFlag = multiprocessing.Value("i", 0)

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
    showwindow = ShowWindow(qFace,qList,qPost,qCapture,qSmallFace,terminateAll)
    recprocess = RecProcess(qList,qPost,loadNetEndFlag)
    captureprocess = CaptureProcess(qCapture,qFace)
    smallface = SmallFace(qFace,qList,qSmallFace)

    recprocess.start()
    while True:
        if loadNetEndFlag.value:
            captureprocess.start()
            smallface.start()
            showwindow.start()
            print "All processes have started."
            break
    while True:
        if terminateAll.value:
            captureprocess.terminate()
            recprocess.terminate()
            smallface.terminate()
            showwindow.terminate()
            del showwindow,recprocess,captureprocess,smallface
            break

'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = MainWindow(qFace, qList,qPost,qCapture)#,playtimerOut,facetimerOut,texttimerOut,databasetimerOut)
    mainwindow.show()
    sys.exit(app.exec_())
'''
