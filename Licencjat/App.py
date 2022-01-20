import os
import shutil
import cv2

if __name__ == "__main__":
    for folder in os.listdir("SplitedVidos"):
        print(folder)
        frame0 = os.listdir(f"SplitedVidos/{folder}")[0]
        f1 = cv2.imread(f"SplitedVidos/{folder}/{frame0}")
        counter = 0
        try:
            shutil.rmtree(f"SplitedDelta/{folder}")
        except FileNotFoundError:
            pass

        os.mkdir(f"SplitedDelta/{folder}")
        a = list(map(lambda x: int(x.split(f"frame")[1].split(".jpg")[0]), os.listdir(f"SplitedVidos/{folder}/")))

        jump = 5
        for i in range(1, len(a), jump):
            print(i)
            f2 = cv2.imread(f"SplitedVidos/{folder}/frame{a[i]}.jpg")
            delta = cv2.subtract(f1, f2)
            cv2.imwrite(f"SplitedDelta/{folder}/delta{counter}.jpg", delta)
            counter += 1
            f1 = f2

