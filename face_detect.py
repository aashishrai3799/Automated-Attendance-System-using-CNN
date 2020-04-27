import cv2
from mtcnn2 import MTCNN
from draw_points import *
import os
import numpy as np

#ckpts = np.zeros((5000, 2500), dtype='uint8')

print('Welcome to Face Detection \n\n Enter 1 to add image manually\n Enter 2 to detect face in Webcam feed')
n = int(input())
if n != 1 and n != 2:
    print('Wrong Choice')
    exit(0)
count = 0
if n == 1:
    print('Enter complete address of the image')
    #addr = str(input())
    #addr = 'C:/Users/Rashmi/Downloads/21.jpg'
    addr = '/home/ml/Documents/attendance_dl/21.jpg'
    if not os.path.exists(addr):
        print('Invalid Address')
        exit(0)

    print('Enter Resolution of output image (in heightXwidth format)')
    res = input().split('X')
    img = cv2.imread(addr)
    img = cv2.resize(img, (int(res[0]), int(res[1])))
    ckpts = np.zeros((int(res[0]), int(res[1])), dtype = 'uint8')

elif n ==2:
    #video_capture = cv2.VideoCapture(0)
    #/home/ml/Documents/attendance_dl/dataset/dtst7.mp4
    video_capture = cv2.VideoCapture('dataset/Mam.mp4')


detector = MTCNN()
ct = 0
alpha = 0.12
beta = 0.04

while True:

    if n == 2:
        ret, frame = video_capture.read()
        #frame = cv2.resize(frame)
    elif n == 1:
        frame = img

    #edges = cv2.Canny(frame,500,1000)
    #b, g, r = cv2.split(frame)
    #dst = cv2.add(r, edges)
    #frame2 = cv2.merge((r, b, dst))
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2+250), -90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    frame = cv2.resize(frame, (840, 480))

    detect = detector.detect_faces(frame)

    if detect:

        for i in range(int(len(detect[:]))):
            boxes = detect[i]['box']
            keypoints = detect[i]['keypoints']
            #print(keypoints['nose'])
            if ckpts[keypoints['nose']] == 0 and ckpts[keypoints['left_eye']] == 0 and ckpts[keypoints['right_eye']] == 0 and ckpts[keypoints['mouth_left']] == 0 and ckpts[keypoints['mouth_right']] == 0:
                #show_points(frame, boxes, keypoints, alpha, beta)
                draw_lines(frame, boxes, keypoints, alpha, beta, count)
                count = count+1
                print('count', count)
                '''for w in range(boxes[0], boxes[0]+boxes[2]):
                    for h in range(boxes[1], boxes[1]+boxes[3]):
                        ckpts[w][h] = 1'''


    # Display the resulting frame
    cv2.imshow('Frame', frame)
    #cv2.waitKey(0)
    #break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
#video_capture.release()
cv2.destroyAllWindows()
