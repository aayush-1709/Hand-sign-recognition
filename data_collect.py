import os
import cv2
import copy
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "log"
os.makedirs(folder, exist_ok=True)


cat_type = input("Enter_category_type: ")
if cat_type.strip():
    os.makedirs(f'{folder}/{cat_type}', exist_ok=True)
    counter = len(os.listdir(f'{folder}/{cat_type}'))

while True:
    success, _img = cap.read()
    __img = copy.deepcopy(_img)
    hands, img = detector.findHands(_img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure the cropping coordinates are within the image dimensions
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Add this check to avoid errors when imgCrop is empty
        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/{cat_type}/Image_{counter}.jpg', __img)#imgWhite)
        print(counter)
    elif key == ord('x'):
        cv2.destroyAllWindows()
        break