import numpy as np
import cv2

cv2.namedWindow("rtmpCap", flags=cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture("rtmp://127.0.0.1:1935/live/ios")

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('rtmpCap',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()