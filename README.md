# This is the official implementation of

## An End-to-End Real-Time Face Identification and Attendance System using Convolutional Neural Networks 
(https://ieeexplore.ieee.org/document/9029001)



An end-to-end face identification and attendance approach using Convolutional Neural Networks (CNN), which processes the CCTV footage or a video of the class and mark the attendance of the entire class simultaneously. One of the main advantages of the proposed solution is its robustness against usual challenges like occlusion (partially visible/covered faces), orientation, alignment and luminescence of the classroom.

# Libraries
1. Tensorflow 1.14
2. Numpy
3. OpenCV
4. MTCNN
5. Sklearn
6. xlsxwriter, xlrd
7. scipy
8. pickle


# How to use

## Installation
1. Install the required libraries. (Conda environment preferred).
2. Download the pre-trained model from [[link]](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) and copy to the main directory.
3. Make sure to have the below mentioned directory structure (you've to manually create two folders named "attendance" and "output" in the main directory | refer to the "Main" directory structure).
4. To verify if everything is installed correctly, run 'user_interface.py'.

## Create Dataset
1. Run 'user_interface.py'
2. Click on the 'Create' button.
3. Select 'webcam' if you wish to create live dataset. (you can leave all other fileds empty)
4. Click on the 'Continue' button to start streaming webcam feed.
5. Press 's' to save the face images. Take as many images as you can take. (approx. 80-100 preferred)
6. Press 'q' to exit.
7. Likewise create other datasets.

## Training
1. Run 'user_interface.py'
2. Click on the 'Train' button.
3. Training may take several minutes (depending upon your system configuration).
4. Once training is completed, a 'classifier.pkl' file will be generated.

## Run
1. Run 'user_interface.py'
2. Click on the 'Run' button.
3. Select 'Webcam' fom the list and leave all fields blank.
4. Click on 'Mark Attendance' button.
5. Attendance sheet will be generated automatically with current date/time.

## Make sure to have following directory structure
1. 'Main' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image5.png" width="480">
2. 'output' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image4.png" width="480">
3. '20180402-114759' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image3.png" width="480">



The file for data augmentation will be uploaded soon.

To know more about the working of the software, refer to our paper.



## Download pre-trained model:
https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-


## Cite
If you find this paper/code userful, consider citing

```
@INPROCEEDINGS{9029001,  
author={Rai, Aashish and Karnani, Rashmi and Chudasama, Vishal and Upla, Kishor},  
booktitle={2019 IEEE 16th India Council International Conference (INDICON)},   
title={An End-to-End Real-Time Face Identification and Attendance System using Convolutional Neural Networks},   
year={2019},  volume={},  number={},  pages={1-4},  
doi={10.1109/INDICON47234.2019.9029001}}
```

## License

The code is available under MIT License. Please read the license terms available at [[Link]](https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/LICENSE)

