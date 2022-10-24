import cv2
from random import randrange as r
#dataset load
trained_data = cv2.CascadeClassifier("face.xml")
#start the webcame use 0 for camera and any "video.mp4" for clip
webcame = cv2.VideoCapture(0)
while True:
     success,frame = webcame.read()
     
     #convert to grey scale
     greyframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
     #detect faces
     frameCoordinates = trained_data.detectMultiScale(greyframe)
     
     # contineusly take frames
     for x,y,w,h in frameCoordinates:
          cv2.rectangle(frame, (x,y), (x+w,y+h),(r(0,255),r(0,255),r(0,255)), 2)
          
     cv2.imshow("face detection", frame)

     key = cv2.waitKey(1)
     if (key ==81 or key == 113):
          break
webcame.release()
