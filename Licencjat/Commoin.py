import cv2
import os

def delta(img1=None, img2=None):
    if img1 is None or img2 is None:
        return
    return img2-img1


def change(folder="Resized/Nowy folder (20)/"):
    for photo in os.listdir(folder):
        print(photo)
        os.rename(f"{folder}/{photo}", f"{folder}/{photo.split('.')[0]}(2).jpg")

change()