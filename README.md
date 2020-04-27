# Automated-Attendance-System-using-CNN

An end-to-end face identification and attendance approach using Convolutional Neural Networks (CNN), which processes the CCTV footage or a video of the class and mark the attendance of the entire class in a single shot. One of the main advantages of the proposed solution is its robustness against usual challenges like occlusion (partially visible/covered faces), orientation, alignment and luminescence of the classroom.

# Research Paper
The implementation is based on the following paper:

https://ieeexplore.ieee.org/document/9029001

# Make sure to have following directory structure
1. 'Main' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image5.png" width="480">
2. 'output' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image4.png" width="480">
3. '20180402-114759' directory:
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image3.png" width="480">
4. Each person's directory will look somewhat like
<img src="https://github.com/aashishrai3799/Automated-Attendance-System-using-CNN/blob/master/images/image1.png" width="480">



# How to use
1. Take 5-10 seconds video of each person/class.
2. Create  face dataset using face_detect.py or user_interface.
3. Augment data using blurr.py.
4. Run user_interface.py, and train the model.
5. Test/Run using user_interface.py.
6. Attendance sheet will be generated automatically with current date/time.



# Download pre-trained model:
https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
