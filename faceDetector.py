import cv2

from random import randrange

trained_face = cv2.CascadeClassifier("haarcascade.xml")

img = cv2.imread("my face.jpeg")

gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face.detectMultiScale(gs_img, 1.1, 4)

#print(face_coordinates)
#cv2.rectangle(img,(430, 180), (430+205,180+205), (255, 0,0), 2)

for i in face_coordinates:
    (x,y,w,h) = i
    cv2.rectangle(img,(x,y), (x+w,y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)

#print(face_coordinates)

cv2.imshow("Ifes Work", img)
cv2.waitKey()

print("Completed!")
