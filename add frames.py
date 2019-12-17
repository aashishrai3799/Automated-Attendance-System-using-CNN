import cv2

addr = '/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/dataset'
video_capture_1 = cv2.VideoCapture(addr + '/' + 'Vamsi' + '.mp4')
video_capture_2 = cv2.VideoCapture(addr + '/' + 'Roshna' + '.mp4')
video_capture_3 = cv2.VideoCapture(addr + '/' + 'Sweety' + '.mp4')
video_capture_4 = cv2.VideoCapture(addr + '/' + 'Deeksha' + '.mp4')
video_capture_5 = cv2.VideoCapture(addr + '/' + 'Sachin' + '.mp4')
video_capture_6 = cv2.VideoCapture(addr + '/' + 'Nikunj' + '.mp4')
video_capture_7 = cv2.VideoCapture(addr + '/' + 'Nishtha' + '.MOV')
video_capture_8 = cv2.VideoCapture(addr + '/' + 'Prashant' + '.mp4')
video_capture_9 = cv2.VideoCapture(addr + '/' + 'Ankit' + '.mp4')
video_capture_10 = cv2.VideoCapture(addr + '/' + 'Apoorv' + '.mp4')
video_capture_11 = cv2.VideoCapture(addr + '/' + 'Sai' + '.mp4')
video_capture_12 = cv2.VideoCapture(addr + '/' + 'Dhruvil' + '.mp4')

#detector = MTCNN()

while True:
    _, frame1 = video_capture_1.read()
    _, frame2 = video_capture_2.read()
    _, frame3 = video_capture_3.read()
    _, frame4 = video_capture_4.read()
    _, frame5 = video_capture_5.read()
    _, frame6 = video_capture_6.read()
    _, frame7 = video_capture_7.read()
    _, frame8 = video_capture_8.read()
    _, frame9 = video_capture_9.read()
    _, frame10 = video_capture_10.read()
    _, frame11 = video_capture_11.read()
    _, frame12 = video_capture_12.read()

    frame = frame1
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 0, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    frame1 = cv2.resize(frame, (333, 500))

    frame = frame2
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -90, 1)
    frame = cv2.warpAffine(frame, m, (600, 500))
    frame2 = cv2.resize(frame, (333, 500))

    frame = frame3
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 0, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    frame3 = cv2.resize(frame, (333, 500))

    frame = frame4
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -90, 1)
    frame = cv2.warpAffine(frame, m, (500, 600))
    frame4 = cv2.resize(frame, (333, 500))

    frame = frame5
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1]-400, frame.shape[0]+400))
    frame5 = cv2.resize(frame, (333, 500))

    frame = frame6
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1]-100, frame.shape[0]+200))
    frame6 = cv2.resize(frame, (333, 500))

    frame = frame7
    m = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), -90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1]-400, frame.shape[0]+400))
    frame7 = cv2.resize(frame, (333, 500))

    final_frame = cv2.resize(frame1, (2000, 1000))
    final_frame[0:500, 0:333] = frame1
    final_frame[0:500, 333:2*333] = frame2
    final_frame[0:500, 2*333:3*333] = frame3
    final_frame[0:500, 3*333:4*333] = frame4
    final_frame[0:500, 4*333:5*333] = frame5
    final_frame[0:500, 5*333:6*333] = frame6
    final_frame[500:2*500, 0*333:1*333] = frame7


    cv2.imshow('final', final_frame)
    cv2.waitKey(0)


