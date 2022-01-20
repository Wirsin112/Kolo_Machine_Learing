import cv2
import os
import shutil

name = "frames3"
try:
    shutil.rmtree(name)
except FileNotFoundError:
    pass

os.mkdir(name)
vidcap = cv2.VideoCapture('vido4.mp4')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(f"{name}/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
