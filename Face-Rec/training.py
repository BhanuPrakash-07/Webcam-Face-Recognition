import numpy as np
import cv2
import os
import faceRecognition as fr
print (fr)

faces,faceID=fr.labels_for_training_data(r'B:\Python\ML\MiniProjects\Face-Rec\imgs') 
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'B:\Python\ML\MiniProjects\Face-Rec\trainingData.yml')

name={0:"PK",1:"RC",2:"BP"} 
cv2.waitKey(0)
cv2.destroyAllWindows
