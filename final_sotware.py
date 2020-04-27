# /home/aashish/Documents/deep_learning/attendance_deep_learning

import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import facenet
import cv2
import sys
import os
import math
import pickle
from sklearn.svm import SVC
from PIL import Image
from face_aligner import FaceAligner
import detect_face
from sheet import mark_present
from mtcnn.mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'






def dataset_creation(parameters):
    path1, webcam, face_dim, gpu, username, vid_path = parameters
    path = ""
    res = ()
    personNo = 1
    folder_name = ""

    path = path1

    if os.path.isdir(path):
        path += '/output'
        if os.path.isdir(path):
            print("Directory already exists. Using it \n")
        else:
            if not os.makedirs(path):
                print("Directory successfully made in: " + path + "\n")

    else:
        if path == "":
            print("Making an output folder in this directory only. \n")
        else:
            print("No such directory exists. Making an output folder in this current code directory only. \n")

        path = 'output'
        if os.path.isdir(path):
            print("Directory already exists. Using it \n")
        else:
            if os.makedirs(path):
                print("error in making directory. \n")
                sys.exit()
            else:
                print("Directory successfully made: " + path + "\n")
    detector = MTCNN()
    res = webcam
    if res == "":
        res = (640, 480)
    else:
        res = tuple(map(int, res.split('x')))

    gpu_fraction = gpu
    if gpu_fraction == "":
        gpu_fraction = 0.8
    else:
        gpu_fraction = round(float(gpu_fraction), 1)

    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.7

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    face_size = face_dim
    if face_size == "":
        face_size = (160, 160)
        print('default face size')
    else:
        face_size = tuple(map(int, face_size.split('x')))
    affine = FaceAligner(desiredLeftEye=(0.33, 0.33), desiredFaceWidth=face_size[0], desiredFaceHeight=face_size[1])

    while True:
        ask = username
        ask = ask.replace(" ", "_")

        if ask == "":
            folder_name = 'person' + str(personNo)
        else:
            folder_name = ask

        personNo += 1
        users_folder = path + "/" + folder_name
        image_no = 1

        if os.path.isdir(users_folder):
            print("Directory already exists. Using it \n")
        else:
            if os.makedirs(users_folder):
                print("error in making directory. \n")
                sys.exit()
            else:
                print("Directory successfully made: " + users_folder + "\n")

        data_type = vid_path
        loop_type = False
        total_frames = 0

        if data_type == "":
            data_type = 0
            loop_type = True

        # Initialize webcam or video
        device = cv2.VideoCapture(data_type)

        # If webcam set resolution
        if data_type == 0:
            device.set(3, res[0])
            device.set(4, res[1])
        else:
            # Finding total number of frames of video.
            total_frames = int(device.get(cv2.CAP_PROP_FRAME_COUNT))
            # Shutting down webcam variable
            loop_type = False

        # Start web cam or start video and start creating dataset by user.
        while loop_type or (total_frames > 0):

            # If video selected dec counter
            if loop_type == False:
                total_frames -= 1

            ret, image = device.read()

            # Run MTCNN and do face detection until 's' keyword is pressed
            if (cv2.waitKey(1) & 0xFF) == ord("s"):

                #bb, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
                detect = detector.detect_faces(image)
                print(detect)

                # See if face is detected
                if detect:
                  bb = detect[0]['box']
                  points = detect[0]['keypoints']
                  print(bb)
                  x, y, w, h = bb
                  aligned_image = image[y:y+h, x:x+w]
                  #aligned_image = affine.align(image, points)
                  image_name = users_folder + "/" + folder_name + "_" + str(image_no).zfill(4) + ".png"
                  cv2.imwrite(image_name, aligned_image)
                  image_no += 1

                  '''
                    for i in range(bb.shape[0]):
                        cv2.rectangle(image, (int(bb[i][0]), int(bb[i][1])), (int(bb[i][2]), int(bb[i][3])), (0, 255, 0), 2)

                    # loop over the (x, y)-coordinates for the facial landmarks
                    # and draw each of them
                    for col in range(points.shape[1]):
                        for i in range(5):
                            cv2.circle(image, (int(points[i][col]), int(points[i + 5][col])), 1, (0, 255, 0), -1)'''

            # Show the output video to user
            cv2.imshow("Output", image)

            # Break this loop if 'q' keyword pressed to go to next user.
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                device.release()
                cv2.destroyAllWindows()
                # break
                abcd = 1
                return abcd

def train(parameters):
    path1, path2, batch, img_dim, gpu, svm_name, split_percent, split_data = parameters

    path = path1  # input("\nEnter the path to the face images directory inside which multiple user folders are present or press ENTER if the default created output folder is present in this code directory only: ")
    if path == "":
        path = 'output'

    gpu_fraction = gpu  # input("\nEnter the gpu memory fraction u want to allocate out of 1 or press ENTER for default 0.8: ").rstrip().lstrip()
    if gpu_fraction == "":
        gpu_fraction = 0.8
    else:
        gpu_fraction = round(float(gpu_fraction), 1)

    model = path2  # input("\nEnter the FOLDER PATH inside which 20180402-114759 FOLDER is present. Press ENTER stating that the FOLDER 20180402-114759 is present in this code directory itself: ").rstrip().lstrip()
    if model == "":
        model = "20180402-114759/20180402-114759.pb"
    else:
        model += "/20180402-114759/20180402-114759.pb"

    batch_size = 90
    ask = batch  # input("\nEnter the batch size of images to process at once OR press ENTER for default 90: ").rstrip().lstrip()
    if ask != "":
        batch_size = int(ask)

    image_size = 160
    ask = img_dim  # input("\nEnter the width_size of face images OR press ENTER for default 160: ").rstrip().lstrip()
    if ask != "":
        image_size = int(ask)

    classifier_filename = svm_name  # input("Enter the output SVM classifier filename OR press ENTER for default name= classifier: ")
    if classifier_filename == "":
        classifier_filename = 'classifier.pkl'
    else:
        classifier_filename += '.pkl'
    classifier_filename = os.path.expanduser(classifier_filename)

    split_dataset = split_data  # input("\nPress Y if you want to split the dataset for Training and Testing: ").rstrip().lstrip().lower()

    # If yes ask for the percentage of training and testing division.
    percentage = 70
    if split_dataset == 'y':
        ask = split_percent  # input("\nEnter the percentage of training dataset for splitting OR press ENTER for default 70: ").rstrip().lstrip()
        if ask != "":
            percentage = float(ask)

    min_nrof_images_per_class = 0

    dataset = facenet.get_dataset(path)
    train_set = []
    test_set = []

    if split_dataset == 'y':
        for cls in dataset:
            paths = cls.image_paths
            # Remove classes with less than min_nrof_images_per_class
            if len(paths) >= min_nrof_images_per_class:
                np.random.shuffle(paths)

                # Find the number of images in training set and testing set images for this class
                no_train_images = int(percentage * len(paths) * 0.01)

                train_set.append(facenet.ImageClass(cls.name, paths[:no_train_images]))
                test_set.append(facenet.ImageClass(cls.name, paths[no_train_images:]))


    paths_train = []
    labels_train = []
    paths_test = []
    labels_test = []
    emb_array = []
    class_names = []

    if split_dataset == 'y':
        paths_train, labels_train = facenet.get_image_paths_and_labels(train_set)
        paths_test, labels_test = facenet.get_image_paths_and_labels(test_set)
        print('\nNumber of classes: %d' % len(train_set))
        print('\nNumber of images in TRAIN set: %d' % len(paths_train))
        print('\nNumber of images in TEST set: %d' % len(paths_test))
    else:
        paths_train, labels_train = facenet.get_image_paths_and_labels(dataset)
        print('\nNumber of classes: %d' % len(dataset))
        print('\nNumber of images: %d' % len(paths_train))

    # Find embedding
    emb_array = get_embeddings(model, paths_train, batch_size, image_size, gpu_fraction)

    # Train the classifier
    print('\nTraining classifier')
    model_svc = SVC(kernel='linear', probability=True)
    model_svc.fit(emb_array, labels_train)

    # Create a list of class names
    if split_dataset == 'y':
        class_names = [cls.name.replace('_', ' ') for cls in train_set]
    else:
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

    # Saving classifier model
    with open(classifier_filename, 'wb') as outfile:
        pickle.dump((model_svc, class_names), outfile)

    print('\nSaved classifier model to file: "%s"' % classifier_filename)

    if split_dataset == 'y':
        # Find embedding for test data
        emb_array = get_embeddings(model, paths_test, batch_size, image_size, gpu_fraction)

        # Call test on the test set.
        parameters = '', '', '', '', '', gpu_fraction
        test(parameters, classifier_filename, emb_array, labels_test, model, batch_size, image_size)

    c = 1
    return c


def test(parameters, classifier_filename="", emb_array=[], labels_test=[], model="", batch_size=0, image_size=0):
    path1, path2, path3, batch_size, img_dim, gpu = parameters

    if classifier_filename == "":
        classifier_filename = path1  # input("\nEnter the path of the classifier .pkl file or press ENTER if a filename classifier.pkl is present in this code directory itself: ")
        if classifier_filename == "":
            classifier_filename = 'classifier.pkl'
        classifier_filename = os.path.expanduser(classifier_filename)

    gpu_fraction = gpu  # input("\nEnter the gpu memory fraction u want to allocate out of 1 or press ENTER for default 0.8: ").rstrip().lstrip()
    if gpu_fraction == "":
        gpu_fraction = 0.8
    else:
        gpu_fraction = round(float(gpu_fraction), 1)

    if model == "":
        model = path2  # input("\nEnter the FOLDER PATH inside which 20180402-114759 FOLDER is present. Press ENTER stating that the FOLDER 20180402-114759 is present in this code directory itself: ").rstrip()
    if model == "":
        model = "20180402-114759/20180402-114759.pb"

    if batch_size == 0 or batch_size == '':
        ask = batch_size  # input("\nEnter the batch size of images to process at once OR press ENTER for default 90: ").rstrip().lstrip()
        if ask == "":
            batch_size = 90
        else:
            batch_size = int(ask)

    if image_size == 0:
        ask = img_dim  # input("\nEnter the width_size of face images OR press ENTER for default 160: ").rstrip().lstrip()
        if ask == "":
            image_size = 160
        else:
            image_size = int(ask)

    if labels_test == []:
        path = path3  # input("\nEnter the path to the face images directory inside which multiple user folders are present or press ENTER if the default created output folder is present in this code directory only: ")
        if path == "":
            path = 'output'
        dataset = facenet.get_dataset(path)
        paths, labels_test = facenet.get_image_paths_and_labels(dataset)
        print('\nNumber of classes to test: %d' % len(dataset))
        print('\nNumber of images to test: %d' % len(paths))
        # Generate embeddings of these paths
        emb_array = get_embeddings(model, paths, batch_size, image_size, gpu_fraction)

    # Classify images
    print('\nTesting classifier')
    with open(classifier_filename, 'rb') as infile:
        (modelSVM, class_names) = pickle.load(infile)

    print('\nLoaded classifier model from file "%s"' % classifier_filename)

    predictions = modelSVM.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

    for i in range(len(best_class_indices)):
        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

    accuracy = np.mean(np.equal(best_class_indices, labels_test))
    print('\nAccuracy: %.3f' % accuracy)


def get_embeddings(model, paths, batch_size, image_size, gpu_fraction):
    # initializing the facenet tensorflow model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # Load the model
            print('\nLoading feature extraction model')
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                # print(paths_batch)

                # Does random crop, prewhitening and flipping.
                images = facenet.load_data(paths_batch, False, False, image_size)

                # Get the embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    return emb_array


def recognize(mode, parameters):
    print(parameters)
    path1, path2, face_dim, gpu, thresh1, thresh2, resolution, img_path, out_img_path, vid_path, vid_save, vid_see = parameters
    st_name = ''
    # Taking the parameters for recogniton by the user
    if path1:
        classifier_filename = path1  # input("\nEnter the path of the classifier .pkl file or press ENTER if a filename 'classifier.pkl' is present in this code directory itself: ")
    else:
        classifier_filename = 'classifier.pkl'
    classifier_filename = os.path.expanduser(classifier_filename)

    if path2:
        model = path2  # input("\nEnter the FOLDER PATH inside which 20180402-114759 FOLDER is present. Press ENTER stating that the FOLDER 20180402-114759 is present in this code directory itself: ").rstrip()
    else:
        model = "20180402-114759/20180402-114759.pb"

    # Create an object of face aligner module
    image_size = (160, 160)
    if face_dim:
        ask = face_dim  # input("\nEnter desired face width and height in WidthxHeight format for face aligner to take OR press ENTER for default 160x160 pixel: ").rstrip().lower()
        image_size = tuple(map(int, ask.split('x')))

    # Take gpu fraction values
    if gpu:
        gpu_fraction = gpu  # input("\nEnter the gpu memory fraction u want to allocate out of 1 or press ENTER for default 0.8: ").rstrip()
        gpu_fraction = round(float(gpu_fraction), 1)

    else:
        gpu_fraction = 0.8

    # input_type = input("\nPress I for image input OR\nPress V for video input OR\nPress W for webcam input OR\nPress ENTER for default webcam: ").lstrip().rstrip().lower()
    # if input_type == "":
    #  input_type = 'w'
    input_type = mode

    # Load the face aligner model
    affine = FaceAligner(desiredLeftEye=(0.33, 0.33), desiredFaceWidth=image_size[0], desiredFaceHeight=image_size[1])

    # Building seperate graphs for both the tf architectures
    g1 = tf.Graph()
    g2 = tf.Graph()

    # Load the model for FaceNet image recognition
    with g1.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with tf.Session() as sess:
            facenet.load_model(model)

    # Load the model of MTCNN face detection.
    with g2.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    # Some MTCNN network parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.8]  # Three steps's threshold
    factor = 0.709  # scale factor
    if thresh1:
        ask = thresh1  # input("\nEnter the threshold FACE DETECTION CONFIDENCE SCORE to consider detection by MTCNN OR press ENTER for default 0.80: ")
        if ask != "" and float(ask) < 1:
            threshold[2] = round(float(ask), 2)

    classifier_threshold = 0.50
    if thresh2:
        ask = thresh2  # input("\nEnter the threshold FACE RECOGNITION CONFIDENCE SCORE to consider face is recognised OR press ENTER for default 0.50: ")
        if ask != "":
            classifier_threshold = float(ask)

    # Loading the classifier model
    with open(classifier_filename, 'rb') as infile:
        (modelSVM, class_names) = pickle.load(infile)

    # helper variables
    image = []
    device = []
    display_output = True

    # Webcam variables
    loop_type = False
    res = (640, 480)

    # Video input variables
    total_frames = 0
    save_video = False
    frame_no = 1
    output_video = []

    # image input type variables
    save_images = False
    image_folder = ""
    out_img_folder = ""
    imageNo = 1
    image_list = []
    image_name = ""

    # If web cam is selected
    if input_type == "w":
        data_type = 0
        loop_type = True
        # Ask for webcam resolution
        if resolution:
            ask = resolution  # input("\nEnter your webcam SUPPORTED resolution for face detection. For eg. 640x480 OR press ENTER for default 640x480: ").rstrip().lower()
            if ask != "":
                res = tuple(map(int, ask.split('x')))

    # If image selected, go to image function.
    elif input_type == "i":

        # Create a list of images inside the given folder
        if img_path:
            image_folder = img_path  # input("\nWrite the folder path inside which images are kept: ").rstrip().lstrip()
        for img in os.listdir(image_folder):
            image_list.append(img)
        total_frames = len(image_list)

        path = 'y'  # vid_save #input("\nIf you want to save the output images to a folder press Y OR press ENTER to ignore it: ").lstrip().rstrip().lower()

        if path == "y":
            save_images = True
            path = out_img_path  # input("\nEnter the location of output folder OR press ENTER to default create an output_images directory here only: ").lstrip().rstrip()
            if os.path.isdir(path) or path == "":
                # User given path is present.
                if path == "":
                    path = "output_images"
                else:
                    path += '/output_images'
                if os.path.isdir(path):
                    print("Directory already exists. Using it \n")
                else:
                    if not os.makedirs(path):
                        print("Directory successfully made in: " + path + "\n")
            else:
                print("Error image folder path. Exiting")
                sys.exit()
            out_img_folder = path + "/"


    # Video is selected
    else:
        data_type = vid_path  # input("\nWrite the video path file to open: ").rstrip().lstrip()
        ask = vid_save  # input("\nPress y to save the output video OR simply press ENTER to ignore it: ").lstrip().rstrip().lower()
        if ask == "y":
            save_video = True

    if input_type != "w":
        ask = vid_see  # input("\nSimply press ENTER to see the output video OR press N to switch off the display: ").lstrip().rstrip().lower()
        if ask != "y":
            display_output = False

    # Initialize webcam or video if no image format
    if input_type != "i":
        device = cv2.VideoCapture(data_type)

    # If webcam set resolution
    if input_type == "w":
        device.set(3, res[0])
        device.set(4, res[1])

    elif input_type == "v":
        # Finding total number of frames of video.
        total_frames = int(device.get(cv2.CAP_PROP_FRAME_COUNT))
        # save video feature.
        if save_video:
            # Finding the file format, size and the fps rate
            fps = device.get(cv2.CAP_PROP_FPS)
            video_format = int(device.get(cv2.CAP_PROP_FOURCC))
            frame_size = (int(device.get(cv2.CAP_PROP_FRAME_WIDTH)), int(device.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # Creating video writer to save the video after process if needed
            output_video = cv2.VideoWriter("/home/ml/Documents/attendance_dl/videos/dslr/Output_" + data_type, video_format, fps, frame_size)

    # Start web cam or start video and start creating dataset by user.
    while loop_type or (frame_no <= total_frames):

        if input_type == "i":
            image = cv2.imread(image_folder + "/" + image_list[frame_no - 1])
        else:
            ret, image = device.read()

        # Run MTCNN model to detect faces
        g2.as_default()
        with tf.Session(graph=g2) as sess:
            # we get the bounding boxes as well as the points for the face
            frame = image
            #/home/ml/Documents/attendance_dl/dataset/test.mp4
            image = cv2.resize(image, (800, 600))


            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            value = 0
            h, s, v = cv2.split(hsv)
            v -= value
            #h -= value
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)                                  ################################################################

            #image = noisy('speckle', image)
            image = np.asarray(image, dtype = 'uint8')

            bb, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

        # See if face is detected
        if bb.shape[0] > 0:

            # ALIGNMENT - use the bounding boxes and facial landmarks points to align images

            # create a numpy array to feed the network
            img_list = []
            images = np.empty([bb.shape[0], image.shape[0], image.shape[1]])

            for col in range(points.shape[1]):
                aligned_image = affine.align(image, points[:, col])
                print(aligned_image)
                print("\n" + str(len(aligned_image)))

                # Prewhiten the image for facenet architecture to give better results
                mean = np.mean(aligned_image)
                std = np.std(aligned_image)
                std_adj = np.maximum(std, 1.0 / np.sqrt(aligned_image.size))
                ready_image = np.multiply(np.subtract(aligned_image, mean), 1 / std_adj)
                img_list.append(ready_image)
                images = np.stack(img_list)

            # EMBEDDINGS: Use the processed aligned images for Facenet embeddings

            g1.as_default()
            with tf.Session(graph=g1) as sess:
                # Run forward pass on FaceNet to get the embeddings
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                embedding = sess.run(embeddings, feed_dict=feed_dict)

            # PREDICTION: use the classifier to predict the most likely class (person).
            predictions = modelSVM.predict_proba(embedding)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            # DRAW: draw bounding boxes, landmarks and predicted names

            if save_video or display_output or save_images:
                for i in range(bb.shape[0]):
                    cv2.rectangle(image, (int(bb[i][0]), int(bb[i][1])), (int(bb[i][2]), int(bb[i][3])), (0, 255, 0), 1)

                    # Put name and probability of detection only if given threshold is crossed
                    if best_class_probabilities[i] > classifier_threshold:
                        cv2.putText(image, class_names[best_class_indices[i]], (int(bb[i][0] + 1), int(bb[i][1]) + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                        print(class_names[best_class_indices[i]])
                        st_name += ','
                        st_name += class_names[best_class_indices[i]]
                        mark_present(st_name)
                        #cv2.waitKey(0)
                        #cv2.putText(image, str(round(best_class_probabilities[i] * 100, 2)) + "%", (int(bb[i][0]), int(bb[i][3]) + 7), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1, cv2.LINE_AA)

                # loop over the (x, y)-coordinates for the facial landmarks
                for col in range(points.shape[1]):
                    for i in range(5):
                        cv2.circle(image, (int(points[i][col]), int(points[i + 5][col])), 1, (0, 255, 0), 1)

        if display_output:
            cv2.imshow("Output", image)
        if save_video:
            output_video.write(image)
        if save_images:
            output_name = out_img_folder + image_list[frame_no - 1]
            # Just taking the initial name of the input image and save in jpg which opencv supports for sure
            # output_name = out_img_folder + image_list[frame_no-1].split(".")[0] + ".jpg"
            cv2.imwrite(output_name, image)

        # If video or images selected dec counter
        if loop_type == False:
            # Display the progress
            print("\nProgress: %.2f" % (100 * frame_no / total_frames) + "%")
            frame_no += 1

        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) == 'q':
            # do a bit of cleanup
            if save_video:
                output_video.release()
            device.release()
            cv2.destroyAllWindows()
            break

    return st_name

  



if __name__ == '__main__':
    main()



