import numpy as np
import cv2
import os

base = 'Resized'
for folder in os.listdir("SplitedVidos/"):
    print(f"{base}/{folder}")
    os.makedirs(f"{base}/{folder}", exist_ok=True)
    for photo in os.listdir(f"SplitedVidos/{folder}/"):
        print(photo)
        img = cv2.imread(f"SplitedVidos/{folder}/{photo}", 1)
        img_stretch = cv2.resize(img, (1920, 1080))
        cv2.imwrite(f"{base}/{folder}/{photo}", img_stretch)
