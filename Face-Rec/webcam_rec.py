import numpy as np
import cv2
import os,time

import faceRecognition as fr
print (fr)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'B:\Python\ML\MiniProjects\Face-Rec\trainingData.yml') 

cap=cv2.VideoCapture(0)

name={0:'PK',1:'RC',2:'BP'}
bath=[1,1,1]
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if(bath[label]):
            fr.put_text(test_img,predicted_name+'\n 20 litres granted',x,y)
        else:
            fr.put_text(test_img,predicted_name+'\n maximum capacity reached',x,y)
        time.sleep(5)
        bath[label]=0

    resized_img=cv2.resize(test_img,(1000,700))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
