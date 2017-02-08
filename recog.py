import cv2
import numpy as np
import sys

cascPath = sys.argv[0]

face_cascade = cv2.CascadeClassifier('/home/abhijay/OpenCV-tmp/opencv-3/data/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/home/abhijay/OpenCV-tmp/opencv-3/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

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
        # roi_gray = gray[y:y+h,x:x+w]
        # roi_color = img[y:y+h,x:x+w]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # eyes = eye_cascade.detectMultiScale(roi_gray,1.1,3)
        # for (ex,ey,ew,eh) in eyes:
        #             cv2.rectangle(roi_color,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    #cv2.putText(roi_color,'eye',(ex-ew,ey-eh),font,0.5,(0,255,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#When everything is done, release the capture
video_capture.release()

cv2.destroyAllWindows()
