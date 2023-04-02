import cv2

from random import randrange

trained_face = cv2.CascadeClassifier("haarcascade.xml")

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    
    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face.detectMultiScale(gs_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2) 


    cv2.imshow("Ifes Work", frame)
    key = cv2.waitKey(1)

    # Press q to release. 
    if key==81 or key==113:
        break

webcam.release()