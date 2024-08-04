
# import tensorflow as tf
# from pathlib import Path
# print(tf.__version__)

# from tensorflow.keras.models import load_model  # Use tensorflow.keras for better compatibility


# Ensure you have the correct path to your model
# model_path = 'model (1).h5'

# Load the model
# model = load_model(model_path)

# Example of making a prediction
# Load and preprocess an image for prediction
# import cv2
# import numpy as np
# img_path = Path('D:\Hand Sign Detection\data\processes\ASL_Dataset\Train\A\10.jpg')
# img = cv2.imread(img_path)
# print(img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (64, 64))
# img = np.expand_dims(img, axis=0) / 255.0  # Normalize

# Predict
# predictions = model.predict(img)
# predicted_class = np.argmax(predictions, axis=1)[0]

# # Class labels mapping
# class_labels = {
#     'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
#     'K': 10, 'L': 11, 'M': 12, 'N': 13, 'Nothing': 14, 'O': 15, 'P': 16, 'Q': 17,
#     'R': 18, 'S': 19, 'Space': 20, 'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X': 25,
#     'Y': 26, 'Z': 27
# }

# # Inverse mapping
# inv_class_labels = {v: k for k, v in class_labels.items()}
# predicted_label = inv_class_labels[predicted_class]

# print(f'Predicted Class Label: {predicted_label}')





import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model

# Specify the image path using a raw string
img_path = Path(r'WIN_20240802_19_27_55_Pro.jpg')
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
img = np.expand_dims(img, axis=0) / 255.0  # Normalize
print(img)
model_path = 'model (1).h5'
model = load_model(model_path)

# Predict
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
inv_class_labels = {v: k for k, v in class_labels.items()}
predicted_label = inv_class_labels[predicted_class]

print(f'Predicted Class Label: {predicted_label}')
