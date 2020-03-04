import cv2
from mtcnn2 import MTCNN
import numpy as  np
from random import randint

color = [0, 255, 255]
color1 = [0, 255, 0]
color2 = [0, 0, 255]
color3 = [255, 0, 255]

def show_points(frame, box, keypoints, alpha, beta, ii, j, p):

    (x, y, w, h) = box
    x, y, w, h = int(x/4), int(y/4), int(w/4), int(h/4)
    dep_y = int(h * alpha)
    dep_x = int(w * beta)

    le = int(np.asarray(int(keypoints['left_eye']), dtype = 'uint8')/p)
    re = int(np.asarray(int(keypoints['right_eye']), dtype = 'uint8')/p)
    lm = int(np.asarray(int(keypoints['mouth_left']), dtype = 'uint8')/p)
    rm = int(np.asarray(int(keypoints['mouth_right']), dtype = 'uint8')/p)
    no = int(np.asarray(int(keypoints['nose']), dtype = 'uint8')/p)

    cv2.circle(frame, (le), 2, color, 2)
    cv2.circle(frame, (re), 2, color, 2)
    cv2.circle(frame, (no), 2, color, 2)
    cv2.circle(frame, (lm), 2, color, 2)
    cv2.circle(frame, (rm), 2, color, 2)
    cv2.circle(frame, (x + int(w/2), y + h + dep_y), 2, color, 2)
    cv2.circle(frame, (x + w + dep_x, y + int(h/2)), 2, color, 2)
    cv2.circle(frame, (x + int(w/2), y-dep_y), 2, color, 2)
    cv2.circle(frame, (x-dep_x, y + int(h/2)), 2, color, 2)
    cv2.circle(frame, (x + w, y), 2, color, 2)
    cv2.circle(frame, (x, y), 2, color, 2)
    cv2.circle(frame, (x + w, y + h), 2, color, 2)
    cv2.circle(frame, (x, y+h), 2, color, 2)


def draw_lines(frame, box, keypoints, alpha, beta, ct):

    (x, y, w, h) = box
    #x, y, w, h = int(x/p)+ii, int(y/p)+j, int(w/p), int(h/p)
    print(x, y, w, h)
    framee = frame[y:y+h, x:x+w]
    framee = cv2.resize(framee, (160, 160))
    #cv2.imshow('frm', framee)
    #cv2.waitKey(0)
    #cv2.imwrite('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/output/Mam/' + str(ct) + '.jpg', framee)
    dep_y = int(h * alpha)
    dep_x = int(w * beta)
    '''
    le = tuple(np.asarray(np.asarray(keypoints['left_eye'], dtype = 'uint8')/p, dtype = 'uint8'))
    re = tuple(np.asarray(np.asarray(keypoints['right_eye'], dtype = 'uint8')/p, dtype = 'uint8'))
    lm = tuple(np.asarray(np.asarray(keypoints['mouth_left'], dtype = 'uint8')/p, dtype = 'uint8'))
    rm = tuple(np.asarray(np.asarray(keypoints['mouth_right'], dtype = 'uint8')/p, dtype = 'uint8'))
    no = tuple(np.asarray(np.asarray(keypoints['nose'], dtype = 'uint8')/p, dtype = 'uint8'))
    #le = tuple(np.asarray(le, dtype = 'uint8'))

    cv2.line(frame, (le), (x + w, y + h), color1, 1)  # from left eye
    cv2.line(frame, (le), (x, y), color1, 1)
    cv2.line(frame, (le), (x, y + h), color1, 1)
    cv2.line(frame, (le), (x + w, y), color1, 1)
    cv2.line(frame, (le), (x - dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (le), (x + int(w / 2), y - dep_y), color1, 1)
    cv2.line(frame, (le), (x + w + dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (le), (x + int(w / 2), y + h + dep_y), color1, 1)

    cv2.line(frame, (re), (x + w, y + h), color1, 1)  # from left_nose
    cv2.line(frame, (re), (x, y), color1, 1)
    cv2.line(frame, (re), (x, y + h), color1, 1)
    cv2.line(frame, (re), (x + w, y), color1, 1)
    cv2.line(frame, (re), (x - dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (re), (x + int(w / 2), y - dep_y), color1, 1)
    cv2.line(frame, (re), (x + w + dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (re), (x + int(w / 2), y + h + dep_y), color1, 1)

    cv2.line(frame, (lm), (x + w, y + h), color1, 1)  # from left_mouth
    cv2.line(frame, (lm), (x, y), color1, 1)
    cv2.line(frame, (lm), (x, y + h), color1, 1)
    cv2.line(frame, (lm), (x + w, y), color1, 1)
    cv2.line(frame, (lm), (x - dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (lm), (x + int(w / 2), y - dep_y), color1, 1)
    cv2.line(frame, (lm), (x + w + 5, y + int(h / 2)), color1, 1)
    cv2.line(frame, (lm), (x + int(w / 2), y + h + dep_y), color1, 1)

    cv2.line(frame, (rm), (x + w, y + h), color1, 1)  # from right_mouth
    cv2.line(frame, (rm), (x, y), color1, 1)
    cv2.line(frame, (rm), (x, y + h), color1, 1)
    cv2.line(frame, (rm), (x + w, y), color1, 1)
    cv2.line(frame, (rm), (x - dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (rm), (x + int(w / 2), y - dep_y), color1, 1)
    cv2.line(frame, (rm), (x + w + dep_x, y + int(h / 2)), color1, 1)
    cv2.line(frame, (rm), (x + int(w / 2), y + h + dep_y), color1, 1)

    cv2.line(frame, (no), (x + w, y + h), (color3), 1)  # from nose
    cv2.line(frame, (no), (x, y), (color3), 1)
    cv2.line(frame, (no), (x, y + h), (color3), 1)
    cv2.line(frame, (no), (x + w, y), (color3), 1)
    cv2.line(frame, (no), (x - dep_x, y + int(h / 2)), (color3), 1)
    cv2.line(frame, (no), (x + int(w / 2), y - dep_y), (color3), 1)
    cv2.line(frame, (no), (x + w + dep_x, y + int(h / 2)), (color3), 1)
    cv2.line(frame, (no), (x + int(w / 2), y + h + dep_y), (color3), 1)

    cv2.line(frame, (x, y), (x + int(w / 2), y - dep_y), (color1), 1)
    cv2.line(frame, (x + int(w / 2), y - dep_y), (x + w, y), (color1), 1)
    cv2.line(frame, (x + w, y), (x + w + dep_x, y + int(h / 2)), (color1), 1)
    cv2.line(frame, (x + w + dep_x, y + int(h / 2)), (x + w, y + h), (color1), 1)
    cv2.line(frame, (x + w, y + h), (x + int(w / 2), y + h + dep_y), (color1), 1)
    cv2.line(frame, (x + int(w / 2), y + h + dep_y), (x, y + h), (color1), 1)
    cv2.line(frame, (x, y + h), (x - dep_x, y + int(h / 2)), (color1), 1)
    cv2.line(frame, (x - dep_x, y + int(h / 2)), (x, y), (color1), 1)

    
    cv2.line(frame, (le), (re), color1, 1)
    cv2.line(frame, (le), (rm), color1, 1)
    cv2.line(frame, (le), (lm), color1, 1)
    cv2.line(frame, (le), (no), color1, 1)
    cv2.line(frame, (re), (rm), color1, 1)
    cv2.line(frame, (re), (lm), color1, 1)
    cv2.line(frame, (re), (no), color1, 1)
    cv2.line(frame, (lm), (rm), color1, 1)
    cv2.line(frame, (lm), (no), color1, 1)
    cv2.line(frame, (rm), (no), color1, 1)

    '''



