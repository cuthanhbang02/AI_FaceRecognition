import cv2
import numpy as np
# import Haar Cascade to detect face
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_color, 1.3, 5)

    if faces == ():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

# loop until collect 100 images of user's face
# for each image, crop and show the image on a window 
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'D:/DADN/TTNT/Smarthome_IoT_project_fe/AI_FaceRecognition/Sample Images/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Crop', face)

    else:
        print("Face Not Found !")
        pass
   
    # press Enter to stop or loop until enough
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("100 images for data collected")

