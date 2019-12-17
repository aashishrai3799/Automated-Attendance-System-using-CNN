import cv2
import numpy as np
import os


def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy


path_from = '/home/ml/Documents/attendance_dl/output1'  # path of input folder
path_to = '/home/ml/Documents/attendance_dl/output/'    # path of output folder
name = os.listdir(path_from)
for i in name:
    if not os.path.exists(path_to + i):
        os.mkdir(path_to + i)
    k = os.listdir(path_from + i)
    p = 1

    for j in k:
        length = 0
        img = cv2.imread(path_from + i + '/' +  j)
        img0 = img
        #print('/home/ml/Documents/attendance_dl/output/' + i + '/' +  j)
        img = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imshow('original', img)
        img1 = cv2.flip(img, 1)
        hsvv = cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)

        h,s,v = cv2.split(hsvv)
        cv2.normalize(v, v, 0, 150, cv2.NORM_MINMAX)
        img2 = cv2.merge((h,s,v+35))
        img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

        h,s,v = cv2.split(hsvv)
        cv2.normalize(v, v, 150, 255, cv2.NORM_MINMAX)
        img4 = cv2.merge((h,s,v-10))
        img4 = cv2.cvtColor(img4, cv2.COLOR_HSV2BGR)

        noisy1 = noisy('gauss', img0)
        noisy2 = noisy('s&p', img0)
        noisy3 = noisy('poisson', img0)
        #noisy4 = noisy('speckle', img0)

        M = np.float32([[1, 0, 40], [0, 1, -5]])
        dst1 = cv2.warpAffine(img0, M, (img0.shape[1], img0.shape[0]))

        M = np.float32([[1, 0, -4], [0, 1, -40]])
        dst = cv2.warpAffine(img0, M, (img0.shape[1], img0.shape[0]))

        cv2.imshow('img1', img2)
        #cv2.waitKey(0)
        cv2.imwrite(path_to + i + '/' + i + '_' + str(length + p) + '.jpg', img1)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+1) + '.jpg', img2)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+2) + '.jpg', img4)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+5) + '.jpg', img0)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+3) + '.jpg', noisy1)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+4) + '.jpg', dst1)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+2) + '.jpg', dst)
        cv2.imwrite(path_to + i + '/' + i  + '_' + str(length + p+7) + '.jpg', noisy4)

        p = p + 8
        if p >= 1000:
            break


'''Deep Learning based automated face attendance system.
1. Process the CCTV footage of the classroom and mark the attendanc of all the students in one go.
2. Make the system robust against usual challenges like occlusion, orientation, alignment, and luminescence of the classroom.
3. Save the attendance of students in a spreadsheet corresponding to the selected date and subject code.
4. Created an Interactive Graphical User Interface.'''

