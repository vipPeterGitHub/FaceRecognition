import dlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
from numpy import *
	
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




	#cv2.imwrite(pic_path,bgrimg)
	#cv2.imshow("frame",bgrimg)
	#cv2.waitKey(0)
	#cv2.imshow("frame",face)
	#cv2.waitKey(0)

'''
	for k, d in enumerate(faces):
		shape = predictor(rgbimg, d)
	for i in range(68):
		pt=shape.part(i)
		plt.plot(pt.x,pt.y,'ro')
		plt.imshow(rgbimg)
		plt.show()
'''

#start = time.time()
if getFace('D:/pys/a.png',detector, predictor)!=False:
	#end = time.time()
	#print end - start
	a,b = getFace('D:/pys/a.png',detector, predictor)
	cv2.imshow("frame",a)
	cv2.waitKey(0)
	cv2.imshow("frame",b)
	cv2.waitKey(0)
#im = cv2.imread('D:/pys/a.png',1)


print 'Hello'
