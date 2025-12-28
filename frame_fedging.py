import cv2
import os
import matplotlib.pyplot as plt

vid = cv2.VideoCapture('vid2.mp4')
current_frame = 0

if not os.path.exists('frames2'):
    os.makedirs('frames2')

while True:
    success, frame2 = vid.read()
    if not success:
        break
    cv2.imshow('Output', frame2)
    cv2.imwrite(os.path.join('frames2', f'{current_frame}.png'), frame2)
    current_frame += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


