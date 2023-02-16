import cv2
import os

# Get the current working directory
cwd = os.getcwd()

# Import the picture to analyse and the xml openCV file for face detection
image_path = "{0}/Apollo_11_Crew.jpg".format(cwd)
cascadeClassifierPath = "{0}/haarcascade_frontalface_alt.xml".format(cwd)

cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)

image = cv2.imread(image_path)

# Turns the image Gray for detection
grayFace = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Does the face detection
detectedFace = cascadeClassifier.detectMultiScale(grayFace)

# Create Green squares around detected faces area of the Gray image version
for(x, y, width, height) in detectedFace:
    cv2.rectangle(image, (x,y), (x+width, y+height), (0,255,0),5)

# Create new image named resultat.jpg with the detected area in green
    cv2.imwrite('resultat.jpg', image)