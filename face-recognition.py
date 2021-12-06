#!/usr/bin/env python
# coding: utf-8

# In[35]:


import face_recognition
import numpy as np
import cv2


video_capture=cv2.VideoCapture(0)

duterte_image=face_recognition.load_image_file("./known_image/Duterte.jpg")
leni_image=face_recognition.load_image_file("./known_image/leni.jpg")
obama_image=face_recognition.load_image_file("./known_image/obama.jpg")
unknown_image=face_recognition.load_image_file("./unknown_image/idk.jpeg")


duterte_face_encoding=face_recognition.face_encodings(duterte_image)[0]
leni_face_encoding=face_recognition.face_encodings(leni_image)[0]
obama_face_encoding=face_recognition.face_encodings(obama_image)[0]
unknown_face_encoding=face_recognition.face_encodings(unknown_image)[0]


known_faces=[duterte_face_encoding, 
              leni_face_encoding ,
              obama_face_encoding]

known_face_names = [
    "Duterte",
    "Leni",
    "obama"
]

results= face_recognition.compare_faces(known_faces, unknown_face_encoding)



print("duterte? {}".format(results[0]))
print("leni? {}".format(results[1]))

print("obama? {}".format(results[2]))

