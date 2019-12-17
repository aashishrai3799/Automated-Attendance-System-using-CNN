import cv2

# Opens the Video file
cap = cv2.VideoCapture('/home/aashish/Documents/deep_learning/dtst/nigam.mp4')
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame,(160,160))
    cv2.imwrite('/home/aashish/Documents/deep_learning/attendance_deep_learning/scripts_used/output/nigam/nigam_' + str(i) + '.jpg', frame)

    i += 1

cap.release()
cv2.destroyAllWindows()
