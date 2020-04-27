# Automated-Attendance-System-using-CNN

An end-to-end face identification and attendance approach using Convolutional Neural Networks (CNN), which processes the CCTV footage or a video of the class and mark the attendance of the entire class simultaneously. One of the main advantages of the proposed solution is its robustness against usual challenges like occlusion (partially visible/covered faces), orientation, alignment and luminescence of the classroom.


# Make sure to have following directory structure
1. 'Main' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image5.png" width="480">
2. 'output' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image4.png" width="480">
3. '20180402-114759' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image3.png" width="480">
4. Each person's directory will look somewhat like
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image1.png" width="480">

# Libraries
1. Tensorflow 1.14
2. Numpy
3. OpenCV
4. MTCNN
5. xlsxwriter, xlrd
6. scipy
7. pickle


# How to use
## Installation
1. Install the required libraries. (Conda environment preferred).
2. Download the pre-trained model from the link given below and copy to the main directory.
3. Make sure to have the afformantioned directory fomat.
4. To verify is everything installed properly run 'user_interface.py'.
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

The file for data augmentation will be uploaded soon.

To know more about the working of the software, refer to our paper.
## Research Paper
The implementation is based on the following paper:
https://ieeexplore.ieee.org/document/9029001


# Download pre-trained model:
https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
