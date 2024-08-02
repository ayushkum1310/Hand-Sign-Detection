

# ASL Hand Sign Classification

This project involves creating a neural network-based model to classify American Sign Language (ASL) hand signs from images. The objective is to develop an effective image classification system capable of accurately identifying ASL alphabets.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Predictions](#predictions)
9. [License](#license)

## Introduction

This project uses deep learning techniques to classify hand signs in American Sign Language. By leveraging a pre-trained VGG16 model and adding custom layers, the system can recognize and classify ASL alphabets from images.

## Dataset

- **Source:** The dataset consists of colored images of hand signs representing different ASL alphabets.
- **Structure:** Images are organized into directories, each corresponding to a specific ASL alphabet class.

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ayushkum1310/Hand-Sign-Detection.git
    cd Hand-Sign-Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare Data:**
   - Place your dataset in the `ASL_Dataset` directory with subdirectories for training and testing images.

2. **Run the Script:**
   - Execute the training script to train the model:
     ```bash
     python .\src\models\train_model.py
     ```

3. **Make Predictions:**
   - Use the trained model to make predictions on new images:
     ```bash
     python .\src\modeld\prediction.py
     ```

## Model Architecture

- **Base Model:** Pre-trained VGG16 (excluding top layers) with weights from ImageNet.
- **Custom Layers:** 
  - Flatten layer to convert features to 1D.
  - Dense layer with 512 units and ReLU activation.
  - Dense layer with 28 units and softmax activation for classification.

## Training

- **Optimizer:** Adam
- **Loss Function:** Sparse categorical cross-entropy
- **Metrics:** Accuracy
- **Epochs:** Specify the number of epochs in the training script.

## Evaluation

The model is evaluated on a separate test dataset to measure its classification accuracy. The evaluation script can be used to assess performance after training.

## Predictions

- **Preprocessing:** Images are resized to 64x64 pixels and normalized before prediction.
- **Output:** The model provides probabilities for each ASL alphabet class, with the highest probability indicating the predicted class.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


