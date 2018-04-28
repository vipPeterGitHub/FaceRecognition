import os
import cv2
import shutil
import time
import dlib


predictor_path = '../Models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def det2face(img, d):
    [x1,x2,y1,y2] = [d.left(),d.right(),d.top(),d.bottom()]
    w = x2-x1
    h = y2-y1
    x1 = x1-w/4
    x2 = x2+w/4
    y1 = y1-w/2
    y2 = y2+w/3
    if x1<0:
      x1 = 10
    if y1<0:
      y1 = 10
    return img[y1:y2,x1:x2,:]

def writeFace(img, det, cnt, facePath):
    img = det2face(img,det)
    img = cv2.resize(img,(300,300))
    cv2.imwrite(facePath+time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())+'-'+str(cnt)+'.jpg',img)


def spiderpic2face():
    spiderpicPath = 'D:/face_recognition/testpy/pachong/'
    facePath = 'pachongface/'
    os.mkdir(facePath)
    cnt = 1
    pics = os.listdir(spiderpicPath)
    for pic in pics:
        picPath = spiderpicPath+pic
        try:
            img = cv2.imread(picPath)
            gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            dets = detector(gray)
            if len(dets)>0:
                for i, det in enumerate(dets):
                    writeFace(img, det, cnt, facePath)
                    print cnt
                    cnt+=1
                    
                    break
        except:
            print "error occurs!"


def movePic():
    rootPath = 'D:/face_recognition/LinClassFaces/'
    resultPath = 'D:/face_recognition/LinClassFaces/allpic/'
    os.mkdir(resultPath)
    cnt = 1
    path1s = os.listdir(rootPath)
    print path1s
    for path1 in path1s:
        path1 = rootPath+path1
        path2s = os.listdir(path1)
        print path2s
        for path2 in path2s:
            path2 = path1+'/'+path2
            path3s = os.listdir(path2)
            print path3s
            for path3 in path3s:
                if (path3.split('.')[1]=='jpg'):
                    path3 = path2+'/'+path3
                    print path3
                    #pathAll = rootPath+path1+path2+path3
                    shutil.copyfile(path3, resultPath+'person'+str(cnt)+'.jpg')
                    print cnt
                    cnt +=1

def movePicVideo():
    rootPath = 'D:/face_recognition/LinClassFaces/'
    picsPath = 'D:/face_recognition/LinClassFaces_allpic/'
    videosPath = 'D:/face_recognition/LinClassFaces_allvideo/'
    os.mkdir(picsPath)
    os.mkdir(videosPath)
    cnt = 1
    path1s = os.listdir(rootPath)
    print path1s
    for path1 in path1s:
        path1 = rootPath+path1
        '''
        path2s = os.listdir(path1)
        print path2s
        for path2 in path2s:
            path2 = path1+'/'+path2
        '''

        path3s = os.listdir(path1)
        print path3s
        for path3 in path3s:
            if (path3.split('.')[1]=='jpg' or path3.split('.')[1]=='JPG' or path3.split('.')[1]=='png'):
                #path3 = path1+'/'+path3
                #print path3
                #pathAll = rootPath+path1+path2+path3
                print path1+'/'+path3
                print picsPath+path3
                shutil.copyfile(path1+'/'+path3, picsPath+path3)

            if (path3.split('_')[-1]!='store' and path3.split('.')[1]!='jpg' and path3.split('.')[1]!='JPG' and path3.split('.')[1]!='png'):
                print path1+'/'+path3
                print picsPath+path3
                shutil.copyfile(path1+'/'+path3, videosPath+path3)

            print cnt
        cnt +=1


def sortDB():
    DBPath = 'D:/face_recognition/lfw/'
    newPath = 'D:/face_recognition/lfw_new/'
    cnt = 0
    #history_path = 'D:/testImage/history/'
    names = os.listdir(DBPath)
    for name in names:
        facesPath = DBPath+name
        faces = os.listdir(facesPath)
        for face in faces:
            resPath = facesPath+'/'+face
            shutil.copyfile(resPath, newPath+face)
            cnt += 1
            print resPath
            print cnt
            #result = cv2.imread(resPath)
            #cv2.imshow('test',result)
            #cv2.waitKey(1000)
        #break
    print names.__len__()

def pics2video():
    #begin = 201
    #end = 400

    for begin in range(0,6000,200):
        end = begin+200
        picsPath = 'D:/face_recognition/lfw_new/'
        videoPath = 'D:/face_recognition/Testdata/'
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")#(*"XVID")
        saveVideoPath = videoPath+str(begin+1)+'-'+str(end) +'.avi'
        out = cv2.VideoWriter(saveVideoPath,fourcc,1,(250,250))#(1920,1080)) # 10 is speed. 640,480
        cnt = 0
        faces = os.listdir(picsPath)
        for face in faces:
            if cnt == end:
                break
            if cnt >= begin:
                img = cv2.imread(picsPath+face)
                out.write(img)
                out.write(img)
                out.write(img)
                out.write(img)
                out.write(img)
            cnt +=1
                #cv2.imshow("test",img)
                #cv2.waitKey(1000)




if __name__ == "__main__":
    print "Hello"
    #sortDB()
    #movePic()
    movePicVideo()


