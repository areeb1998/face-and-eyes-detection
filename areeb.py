import cv2
import numpy as np
#Loads a classifier from a file.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    #convert pic to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    #face cascade contains numbers of trained images of faces so it uses those images to detect your face
    faces = face_cascade.detectMultiScale(gray, 1.1 , 4)
    for (x, y, h, w) in faces:
        cv2.rectangle(img, (x, y),(x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
          cv2.rectangle(roi_color, (ex, ey) , (ex+ew , ey+eh) , (0, 255, 0) , 2)





    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break

capture.release()
cv2.destroyAllWindows()