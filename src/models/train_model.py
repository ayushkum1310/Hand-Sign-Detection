import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import cv2
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path


import dagshub
dagshub.init(repo_owner='ayushkum1310', repo_name='Hand-Sign-Detection', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



import sys
@dataclass
class Modeltrainerconfig:
    train_model_path=os.path.join('D:\Hand Sign Detection\models','model.h5')
    


# class Model_trainer:
#     def __init__(self):
#         self.model_path_config_obj=Modeltrainerconfig()
#     def initiate_model_trainer(self):
#         try:
#             logging.info("Model trainder has started")
#             # Set directory paths
#             base_path = Path("data\processes\ASL_Dataset")
#             train_dir = os.path.join(base_path, "Train")
#             test_dir = os.path.join(base_path, "Test")

#             # Data augmentation and rescaling
#             train_datagen = ImageDataGenerator(
#                 rescale=1./255,
#                 rotation_range=20,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 horizontal_flip=True,
#                 fill_mode='nearest'
#             )

#             test_datagen = ImageDataGenerator(rescale=1./255)

#             # Load data with integer labels
#             train_generator = train_datagen.flow_from_directory(
#                 train_dir,
#                 target_size=(64, 64),
#                 batch_size=32,
#                 class_mode='sparse'  # Changed to 'sparse'
#             )

#             test_generator = test_datagen.flow_from_directory(
#                 test_dir,
#                 target_size=(64, 64),
#                 batch_size=32,
#                 class_mode='sparse'  # Changed to 'sparse'
#             )


#             base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

#             # Add custom layers on top
#             x = base_model.output
#             x = Flatten()(x)
#             x = Dense(512, activation='relu')(x)
#             predictions = Dense(28, activation='softmax')(x)  # 28 classes

#             # Final model
#             model = Model(inputs=base_model.input, outputs=predictions)

#             # Freeze base model layers
#             for layer in base_model.layers:
#                 layer.trainable = False

#             # Compile model with sparse categorical crossentropy
#             model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#             # Train the model
#             history = model.fit(
#                 train_generator,
#                 steps_per_epoch=train_generator.samples // train_generator.batch_size,
#                 validation_data=test_generator,
#                 validation_steps=test_generator.samples // test_generator.batch_size,
#                 epochs=20
#             )

#             # Save the model
#             model.save('/content/model.h5')
#         except Exception as e:
#             raise CustomException(e,sys)