
# Sign Language Digits Classification

This project implements and compares two deep learning architectures for classifying images of sign language digits (0-9): a custom Convolutional Neural Network (CNN) trained from scratch, and a transfer learning approach leveraging a pre-trained MobileNetV2 network.

Additionally, the project analyzes how model accuracy scales when training data is restricted to small fractions of the original training set.

---

## Project Overview

- **Dataset**: Sign Language Digits Dataset containing grayscale images representing hand gesture digits from 0 to 9.
- **Input Preprocessing**: Raw $64 \times 64$ grayscale images are converted into 3-channel (RGB) inputs to satisfy compatibility requirements for pre-trained feature extractors.
- **Partitions**: Split into 70% Training, 15% Validation, and 15% Test.

---

## Model Architectures

### 1. Custom CNN
Built from scratch using the TensorFlow/Keras sequential API:
- **3 Convolutional Blocks**: Each block utilizes `Conv2D` layers with batch normalization and max-pooling (filters: 32, 64, 128).
- **Regularization**: Dropout layers (0.3 and 0.5) to mitigate overfitting.
- **Classification Head**: Flatten layer, dense hidden layer with 256 units (ReLU activation), and a 10-unit dense output layer (softmax activation).

### 2. Transfer Learning (MobileNetV2)
- **Base Model**: MobileNetV2 pre-trained on ImageNet (weights frozen).
- **Classification Head**: Global Average Pooling, followed by a dense hidden layer (128 units, dropout of 0.5, ReLU), and a 10-unit output layer (softmax activation).

---

## Performance Analysis and Results

To analyze how training data size affects model learning and generalization, both models were trained on 25%, 50%, and 100% of the training dataset:

| Percentage of Training Data Used | Custom CNN Test Accuracy | Transfer Learning Test Accuracy |
| :---: | :---: | :---: |
| **25%** (~360 samples) | ~10.00% (Random Guessing) | **~64.19%** |
| **50%** (~721 samples) | ~59.35% | **~70.97%** |
| **100%** (1443 samples) | ~73.55% | **~77.42%** |

### Key Takeaways
- **Data Constrained Performance**: Under extreme data constraints (25% training data), the custom CNN is unable to extract sufficient features, yielding random performance (~10%). Conversely, MobileNetV2 leverages pre-trained ImageNet weights to maintain robust feature extraction, achieving 64.19% accuracy.
- **Data Scaling**: As data scaling reaches 100%, the performance gap narrows significantly, though the transfer learning model continues to outperform the custom CNN across all data regimes.

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```

2. Ensure the `Hand_digits.zip` dataset is located in the root workspace directory.

3. Install necessary dependencies:
   ```bash
   pip install tensorflow numpy matplotlib pandas seaborn scikit-learn
   ```

4. Open and execute the notebook cells to train the models and reproduce the comparative analysis plots.
```
