# Animal Classification with CNN

This project involves training a Convolutional Neural Network (CNN) to classify animal images into 50 different classes using the **AwA2** dataset (Animals with Attributes 2). The project includes data augmentation techniques and CNN model design in PyTorch.

## Project Structure

- **Data Preprocessing**: The images from the AwA2 dataset are resized, augmented with techniques like salt-and-pepper noise, light rotation, and random affine transformations. The dataset is then split into training and testing sets.
  
- **CNN Model**: A convolutional neural network is designed with three convolutional layers, followed by fully connected layers. The model is trained using the Adam optimizer and cross-entropy loss.

- **Training and Evaluation**: The model is trained on the augmented dataset and evaluated on test data. Training and testing accuracies are logged for each epoch.

## Dataset

The project uses the **Animals with Attributes 2 (AwA2)** dataset. This dataset consists of images of animals from 50 different classes, each with 1000 images. The images are resized to 128x128 for training and testing.

## Data Augmentation

To improve the model's generalization capabilities, several data augmentation techniques are applied:
- **Salt-and-Pepper Noise**: Adds noise to the images by randomly turning some pixels white or black.
- **Light Rotation**: Rotates the image by a random angle between -10° and 10°.
- **Random Horizontal and Vertical Flip**: Randomly flips the image horizontally or vertically.
- **Random Affine Transformations**: Includes rotation, translation, and scaling.
- **Color Jitter**: Randomly adjusts the brightness and contrast of the images.

## Model Architecture

The model consists of the following layers:
- **Convolutional Layers**: Three convolutional layers with increasing filter sizes.
- **Pooling Layers**: Max pooling layers after each convolutional layer to reduce spatial dimensions.
- **Fully Connected Layers**: Two fully connected layers to classify the images into 50 different classes.

### CNN Architecture:
- Conv Layer 1: 3 input channels → 32 output channels, kernel size 3x3
- Conv Layer 2: 32 input channels → 64 output channels, kernel size 3x3
- Conv Layer 3: 64 input channels → 128 output channels, kernel size 3x3
- Fully Connected Layer 1: 128 * 16 * 16 → 512 neurons
- Fully Connected Layer 2: 512 neurons → 50 output classes

## Training

The model is trained using:
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Cross-Entropy Loss for multi-class classification.
- **Batch Size**: 32
- **Epochs**: 20

The model's performance is evaluated by calculating accuracy on both training and test datasets.

## Results

During the training, both training and test accuracies are logged for each epoch, helping us monitor the model's progress and evaluate its generalization performance on unseen data.

## Setup and Usage

To run this project, clone this repository and ensure the required libraries are installed. You can install the necessary dependencies using: