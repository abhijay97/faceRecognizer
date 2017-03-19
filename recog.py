import cv2
import numpy as np
import sys

cascPath = sys.argv[0]

face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

cap = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load('recogniser/trainningData.xml')
id =0 
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while True:

    #
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.3,5
    	)


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
        
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()

cv2.destroyAllWindows()
