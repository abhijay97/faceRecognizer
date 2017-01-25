import os
import cv2
import numpy as np 
from PIL import Image

recogniser = cv2.createFisherFaceRecognizer()
path = 'dataSet'

def getImages(path):
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []
	for imagePath in imagePaths:
		faceImg = Image.open(imagePath).convert('L')
		faceNp = np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow("train",faceNp)
		cv2.waitKey(10)

	return np.array(IDs),faces
Ids,faces = getImages(path)
recogniser.train(faces,Ids)
recogniser.save('recogniser/trainningData.xml')
cv2.destroyAllWindow()