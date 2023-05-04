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

face_classifier = cv2.CascadeClassifier('C:/Users/asus/Downloads/Face-Unlock-and-Lock-Recogination-System-main/Face-Unlock-and-Lock-Recogination-System-main/haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces == ():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        roi = img[y: y+h, x: x+w]
        roi = cv2.resize(roi, (200,200))
    return img, roi

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1- (result[1]) / 300))
            display_string = str(confidence) + '% Accuracy'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX,1,(255,120,255),2)

        if confidence > 80:
            cv2.putText(image,"Unlocked", (250, 450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
            cv2.imshow("Face Crop", image)

        else:
            cv2.putText(image,"Locked", (250,450),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Face Crop", image)

    except:
        cv2.putText(image,"Face Not Found",(250,450), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.imshow("Face Crop", image)
        pass

    # press Enter to stop or loop until enough
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()