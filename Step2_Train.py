# can skip this file to step 3

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'D:/DADN/TTNT/Smarthome_IoT_project_fe/AI_FaceRecognition/Sample Images/'
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_data, Labels = [], []

# read dataset folder and make 2 arrays of training data and labels
for i, files in enumerate(only_files):
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images, dtype= np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype= np.int32)

# create model with LBPH algorithm by OpenCV
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data), np.asarray(Labels))

print("Model Trained!")