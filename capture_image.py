import numpy as np
import cv2
import time
import os
from time import sleep

# Label your name here
label = "Thao"

cap = cv2.VideoCapture(0)

# Variable for counting, saving data after the first 100 frames
i=0
while(True):
    # Capture frame by frame
    i+=1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4)

    # Show frame
    cv2.imshow('frame',frame)

    # Save data to folder
    if i>=200 and i<=250: #capture 1000 frames
        print("Number of photos capture = ",i-200)
        # Create folder con
        if not os.path.exists('data' + str(label)):
            os.mkdir('data' + str(label))

        cv2.imwrite('data' + str(label) + "/" + str(i) + ".png",frame)
        sleep(2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()