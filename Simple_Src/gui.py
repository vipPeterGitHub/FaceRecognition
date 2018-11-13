# -*- coding: UTF-8 -*-
import Tkinter as tk
from Tkinter import *
import os
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil 
import time
from PIL import Image, ImageTk
import cv2
import threading
import multiprocessing
import dlib

plt.ion()

dlib_predictor_path = 'D:/pys/shape_predictor_68_face_landmarks.dat'
global detector,predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_predictor_path)

def getFace(pic_path, detector, predictor):
    
    bgrimg = cv2.imread(pic_path)
    
    #print bgrimg.shape
    #rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)
    faces = detector(bgrimg, 1)
    l = len(faces)
    print(" %d face(s) is(are) detected."%(l))
    
    if l > 0:
        face=max(faces, key=lambda rect: rect.width() * rect.height())
        [x1,x2,y1,y2]=[face.left(),face.right(),face.top(),face.bottom()]
        print x1,x2,y1,y2
        e = 8
        face = bgrimg[y1-e:y2+e,x1-e:x2+e]
        e = e+2
        cv2.rectangle(bgrimg,(x1-e,y1-e),(x2+e,y2+e),(0,0,255),2)
        return bgrimg,face
    else:
        print 'No face!'
        return False



def showImage(path):
    lena = mpimg.imread(path)
    #lena.shape #(512, 512, 3)
    plt.imshow(lena)
    plt.axis('on')
    plt.show()
    plt.pause(2)
    plt.close()

image_path = 'D:\\IDcard_test_program\\'
history_path = 'D:\\testImage\\history\\'

def runDisplay():
    #global tkimg
    #global im
    global bm
    while(True):
        time.sleep(0.5)
        image_name = os.listdir(image_path)
        if image_name.__len__()==14:
            pic_path = image_path + image_name[13]
            if not image_name:
                print ('Image folder is empty!')
            else:
                #showImage(pic_path)
                #im = Image.open(pic_path)
                #tkimg = ImageTk.PhotoImage(im)
                bm = ImageTk.PhotoImage(file = pic_path)
                if not os.path.exists(history_path+image_name[13]):
                    shutil.move(pic_path,"D:\\testImage\\history")
                    print (pic_path+' has been move to history.')
                else:
                    os.remove(pic_path)
                    print (pic_path+' has been deleted.')
        else:
            #im = Image.open('D:/pys/nopic.jpg')
            #tkimg = ImageTk.PhotoImage(im)
            bm = ImageTk.PhotoImage(file = 'D:/pys/nopic.jpg')

        label = tk.Label(image = bm, bg = 'white', width = 112,height = 136)
        label.place(x = 580, y = 100)
        time.sleep(1)

    #cv = Canvas(window,bg = 'white',width=108,height=132)
    #cv.create_image(56,68,image = tkimg)
    #cv.place(x = 550, y = 120)



window = tk.Tk()
window.title('Face Recognition Program')
window.geometry('800x600')


hit = True
def hitAction():
    global hit
    if hit == True:
        hit = False
        var.set('You hit me.')
        os.system("python D:/pys/hello.py")
        print (hit)
    else:
        hit = True
        var.set('No words.')
        os.system("python D:/pys/nihao.py")
        print (hit)
'''
var = tk.StringVar()
l = tk.Label(window, 
    textvariable=var,
    bg='green', font=('Arial', 12), width=15, height=2)
l.place(x=300,y=550)

b = tk.Button(window, text = 'hit me', bg = 'blue', font=('Arial', 12),
              width = 12, height = 2,
              command = hitAction)
b.place(x = 300, y = 500)
'''

def cameraCap():
    global imopen
    global cvimg
    global a
    global b
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)   #640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)  #480
    print cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(True):
        ret,frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('a.png',frame)
        if getFace('D:/pys/a.png',detector, predictor)!=False:       
            a,b = getFace('D:/pys/a.png',detector, predictor)
            cv2.imwrite('a.png',a)
            cv2.imwrite('b.png',b)
            bmm = ImageTk.PhotoImage(file = 'D:/pys/b.png')

            #cv2.imshow("frame",gray)
            #imopen = Image.open('D:/pys/a.png')
            #cvimg = ImageTk.PhotoImage(imopen)
            #cv = Canvas(window,bg = 'white',width=200,height=200)
            #cv.create_image(10,10,image = cvimg)
            #cv.place(x = 50, y = 120)
        else:
            bmm = ImageTk.PhotoImage(file = 'D:/pys/nopic.jpg')
            
        #bm = ImageTk.PhotoImage(file = 'D:/pys/a.png')
        #
        label = tk.Label(image = bmm, width = 112,height = 136)
        label.place(x = 580, y = 285)

        bm = ImageTk.PhotoImage(file = 'D:/pys/a.png')
        label = tk.Label(image = bm, width = 480,height = 360)
        label.place(x = 10, y = 60)
        #time.sleep(0.1)



        #if cv2.waitKey(1)&0xFF == ord('q'):
            #break
    #cap.release()
    #cv2.destroyAllWindows()

cameraCapture = threading.Thread(target=cameraCap,args=(),name='Camera Capture') 
cameraCapture.setDaemon(True) 
cameraCapture.start()

ID_pic = threading.Thread(target=runDisplay,args=(),name='ID card picture') 
ID_pic.setDaemon(True) 
ID_pic.start()

title = tk.Label(window, text = 'Face Recognition Program',
        bg = 'cyan', font = ('Arial', 15), 
        width = 35, height = 2)
title.pack()

def Quit():
    print 'Window will be closed......'
    cameraCapture.sleep(2000)
    cameraCapture.stop()
    #sys.exit()

quit = tk.Button(window, text = 'Quit', bg = 'red', font=('Arial', 12),
              width = 12, height = 2,
              command = window.quit)
quit.place(x = 550, y = 500)

'''
load = tk.Button(window, text = 'Read Image', bg = 'green', font=('Arial', 12),
              width = 12, height = 2,
              command = runDisplay)
load.place(x = 550, y = 400)

camera = tk.Button(window, text = 'Camera', bg = 'green', font=('Arial', 12),
              width = 12, height = 2,
              command = cameraCap)
camera.place(x = 150, y = 400)
'''


l_card_pic = tk.Label(window, text = 'ID Card Pic:', font=('Arial', 12), width=10, height=1)
l_card_pic.place(x=510,y=70)

l_camera_pic = tk.Label(window, text = 'Camera Face:', font=('Arial', 12), width=10, height=1)
l_camera_pic.place(x=510,y=260)


window.mainloop()
