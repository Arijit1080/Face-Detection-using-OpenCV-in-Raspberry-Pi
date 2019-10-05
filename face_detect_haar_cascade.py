#wget -L https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
from imutils import face_utils
import cv2
from imutils.video import VideoStream
import imutils
import time

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("->Starting Face Detection")
c = VideoStream(src=1).start()                   #For webcam, comment it if using Raspberry Pi Camera module
#c = VideoStream(usePiCamera=True).start()       #For Raspberry Pi Camera module, comment it if using webcam
time.sleep(2.0)

while True:

    frame = c.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for rect in rects:
        (x1,y1,w,h) = rect
        x2 = x1 + w
        y2 = y1 + h
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
c.stop()