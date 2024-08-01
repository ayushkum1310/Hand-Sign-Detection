import os
import numpy as np
import os
import sys
# Set the Keras backend to PlaidML
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import plaidml.keras
plaidml.keras.install_backend()
import cv2
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.applications import VGG16

# Set base path and directories
base_path = r"data\processes\ASL_Dataset"  # Use raw string for Windows paths
train_dir = os.path.join(base_path, "Train")
test_dir = os.path.join(base_path, "Test")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=160,
    class_mode='sparse'  # Use 'sparse' for integer labels
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=160,
    class_mode='sparse'
)

# Model setup
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  # Use number of classes

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(train_generator.class_indices)
# Train the model
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_data=test_generator,
#     validation_steps=test_generator.samples // test_generator.batch_size,
#     epochs=1
# )

# Save the model
# model.save(r'C:\path\to\save\model.h5')  # Use raw string for Windows paths

# # Load the model
# model = load_model(r'C:\path\to\save\model.h5')

# # Example prediction
# img_path = r'C:\path\to\image.jpg'  # Use raw string for Windows paths
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (64, 64))
# img = np.expand_dims(img, axis=0) / 255.0  # Normalize

# # Predict
# predictions = model.predict(img)
# predicted_class = np.argmax(predictions, axis=1)
# class_labels = train_generator.class_indices
# inv_class_labels = {v: k for k, v in class_labels.items()}
# predicted_label = inv_class_labels[predicted_class[0]]

# print(f'Predicted Class Label: {predicted_label}')
