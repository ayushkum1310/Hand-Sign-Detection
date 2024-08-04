import os
import sys
from src.logger import logging
from src.exception import CustomException
import cv2
import numpy as np
from pathlib import Path
# import tensorflow as tf
from tensorflow.keras.models import load_model



class Predictor:
    def __init__(self):
        pass
    def Make_prediction(self,img_path:Path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize
        model_path = 'models\model (1).h5'
        model = load_model(model_path)     
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Class labels mapping
        class_labels = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'Nothing': 14, 'O': 15, 'P': 16, 'Q': 17,
            'R': 18, 'S': 19, 'Space': 20, 'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X': 25,
            'Y': 26, 'Z': 27
        }

        # Inverse mapping
        inv_class_labels:dict = {v: k for k, v in class_labels.items()}
        predicted_label:str = inv_class_labels[predicted_class]

        return predicted_label


if __name__=="__main__":
    a=Predictor().Make_prediction(Path("data/processes/ASL_Dataset/Test/A/3001.jpg"))
    print(a)