import cv2
from mtcnn2 import MTCNN
from draw_points import *
import os, numpy as np

print('Welcome to Face Detection \n\n Enter 1 to add image manually\n Enter 2 to detect face in Webcam feed')
#n = int(input())
n=2
if n != 1 and n != 2:
    print('Wrong Choice')
    exit(0)

if n == 1:
    print('Enter complete address of the image')
    #addr = str(input())
    addr = '/home/aashish/Downloads/21.jpg'
    if not os.path.exists(addr):
        print('Invalid Address')
        exit(0)
    print('Enter Resolution of output image (in heightXwidth format)')
    res = input()
    if not res:
        print('default')
        res = '1500X400'

    res = res.split('X')
    img = cv2.imread(addr)
    img = cv2.resize(img, (int(res[0]), int(res[1])))
elif n ==2:
    #video_capture = cv2.VideoCapture(0)
    #/media/aashish/5559-807E/video/class.mp4
    #/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/video/uri2.webm
    video_capture = cv2.VideoCapture('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/video/uri2.webm')
    ckpts = np.zeros((5000, 2500), dtype = 'uint8')
    p = 4
detector = MTCNN()
ct = 0
alpha = 0.12
beta = 0.04
count = 0
while True:
  ret, frame = video_capture.read()
  frame = cv2.resize(frame, (1500, 800))
  #ckpts = np.zeros((5000, 2500), dtype='uint8')

  count = count + 1
  if count % 1 == 0:


    frame2 = frame
    print('frame2', frame2.shape)
    print(frame2.dtype)
    ii, j = 0, 0

    detect = detector.detect_faces(frame)

    if detect:

        for i in range(int(len(detect[:]))):
            boxes = detect[i]['box']
            keypoints = detect[i]['keypoints']
            #print(keypoints['nose'])
            if ckpts[keypoints['nose']] == 0 and ckpts[keypoints['left_eye']] == 0 and ckpts[keypoints['right_eye']] == 0 and ckpts[keypoints['mouth_left']] == 0 and ckpts[keypoints['mouth_right']] == 0:
                #show_points(frame, boxes, keypoints, alpha, beta, ii, j, p)
                draw_lines(frame2, boxes, keypoints, alpha, beta)
                ct = ct+1
                print('ct', ct)
                for w in range(boxes[0], boxes[0]+boxes[2]):
                    for h in range(boxes[1], boxes[1]+boxes[3]):
                        ckpts[w][h] = 1


        #cv2.waitKey(0)

        # Display the resulting frameqq
        cv2.imshow('Frame', frame2)
        #cv2.waitKey(0)
        #break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# Release the capture
#video_capture.release()q
cv2.destroyAllWindows()