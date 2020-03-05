import cv2
name = 'Dhruvil'

# Opens the Video file
cap = cv2.VideoCapture('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/dataset/Mam.mp4')
i = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    '''
    m = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), -90, 1)
    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    frame = cv2.resize(frame, (130, 160))
    padded = cv2.copyMakeBorder(frame, 0, 0, 15, 15, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #frame = frame[0:1100,:]
    '''
    cv2.imshow('frm', frame)


    #frame = cv2.resize(frame,(260,160))
    cv2.imwrite('/home/ml/Documents/attendance_dl/videos/dslr/output_images/' + str(i) + '.jpg', frame)

    i += 1

cap.release()
cv2.destroyAllWindows()
