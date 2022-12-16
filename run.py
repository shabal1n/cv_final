import keras
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from cvzone.HandTrackingModule import HandDetector
import mediapipe
import imutils
import os

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y'] #alphabet
model = keras.models.load_model("sign_language") #model


def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
while True:
    ret, img = cap.read()
    hands = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        top, right, bottom, left = 75, 350, 300, 590
        x, y, w, h = hand['bbox']
        # img = cv2.flip(img, 1)
        roi = img[y: y + 225 + offset, x - offset: x + 290 + offset]
        if len(roi) > 1 and roi[1].size != 0:
            # roi = cv2.flip(roi, 1)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            cv2.imshow('roi', gray)
            alpha = classify(gray)
            cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, alpha, (0, 130), font, 5, (0, 0, 255), 2)
            # cv2.resize(img, (1000, 1000))
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
