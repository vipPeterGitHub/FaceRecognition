# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import dlib


predictor_path = '../../Models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def dlibDetect():
    ttotal = 0
    img = cv2.imread('oscar1.jpg')
    begin = time.clock()
    dets = detector(img,1)
    end = time.clock()
    dtime = end - begin
    ttotal += dtime
    print ("Dlib detection bound time is {}".format(dtime))
    print (len(dets))
    if(len(dets)==0):
        print "No face detected"
    draw = img.copy()
    for k, d in enumerate(dets):
        cv2.rectangle(draw,(d.left(),d.top()),(d.right(),d.bottom()),(255, 255, 0),2)
        begin1 = time.clock()
        shape = predictor(img, d)
        end1 = time.clock()
        ptime = end1 - begin1
        print ptime
        ttotal += ptime
        print len(shape.parts())
        for i, pt in enumerate(shape.parts()):
            #print('Part {}: {}'.format(i, pt))
            pass
            #pt_pos = (pt.x, pt.y)
            #cv2.circle(draw, pt_pos, 3/4, (255, 255, 0), 2)
    print ("Dlib total time is {}".format(ttotal))
    cv2.imwrite("mtcnnRes/"+"Lenna_dlib_label_nomarks1.png",draw)
    cv2.imshow("detection result", draw)
    cv2.waitKey(0)

def mecnnDetect():
    detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = True, threshold=[0.8,0.8,0.9])
    img = cv2.imread('physicists.jpg')
    # # run detector
    begin = time.clock()
    results = detector.detect_face(img)
    end = time.clock()
    print (end - begin)
    print (len(results[0]))
    if results is not None:
        total_boxes = results[0]
        points = results[1]
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 300, 0.37)
        for i, chip in enumerate(chips):
            cv2.imshow('chip_'+str(i), chip)
            cv2.imwrite("mtcnnRes/"+'tennis'+str(i)+'.png', chip)
            #pass
        draw = img.copy()
        for b in total_boxes[:147]:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 2)
        for p in points[:147]:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 255, 0), 2)#red (0,0, 255)
                pass
        cv2.imwrite("mtcnnRes/"+"Lenna_mtcnn_label_marks.png",draw)
        cv2.imshow("detection result", draw)
        cv2.waitKey(0)
        


if __name__ ==  '__main__':
    print 'start...'
    dlibDetect()
    #mecnnDetect()
    print "finished!"