import cv2
import numpy as np
import sys

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global frame
        hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        color = frame[y, x]
        print(x, y)
        print("rgb : ", color)
        color_hsv = hsv[y, x]
        print("hsv : ", color_hsv)


video = cv2.VideoCapture('kanai.mp4')
video.set(cv2.CAP_PROP_FPS, 60)
print("FPS = ", video.get(cv2.CAP_PROP_FPS))
while True:
    ret, frame = video.read()
    if ret:
        cv2.imshow("tracking", frame)
        cv2.setMouseCallback("tracking", onMouse)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = (30, 0, 0)
        upper = (45, 90, 140)

        #lower = (30, 0, 0)
        #upper = (45, 255, 255)

        bin_img = ~cv2.inRange(hsv, lower, upper)
        cv2.imshow("bin", bin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()