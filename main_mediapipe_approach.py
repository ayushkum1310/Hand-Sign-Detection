import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opened.")
    exit()

cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Space", "T", "U", "V", "W", "X", "Y", "Z"]


while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to grab frame.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    imgCrop = np.zeros((imgSize, imgSize, 3), np.uint8)
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
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

        # Get prediction
        try:
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

            # Draw bounding box and label
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)
        except Exception as e:
            print(f"Prediction error: {e}")

    # cv2.imshow("ImageCrop", imgCrop)
    # cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
