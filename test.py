import keras
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from cvzone.HandTrackingModule import HandDetector

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y']
model = keras.models.load_model("sign_language")


def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)
    idx = np.argmax(proba)
    if idx >= 20:
        return alphabet[idx]
    else:
        return ""


def run_ml(file_url):
    cap = cv2.VideoCapture("WA.mp4")
    detector = HandDetector(maxHands=1)

    offset = 20
    counter = 0
    result = ""
    while cap.isOpened():
        ret, img = cap.read()
        if img is None:
            break
        hands = detector.findHands(img, draw=False)
        if hands:
            counter += 1
            hand = hands[0]
            x, y, w, h = hand['bbox']
            roi = img[y: y + 225 + offset, x: x + 290 + offset]
            if len(roi) > 1 and roi[1].size != 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                alpha = classify(gray)
                if counter % 20 == 0:
                    result += alpha
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    print(result)
    # cv2.destroyAllWindows()


# run_ml("WA.mp4")