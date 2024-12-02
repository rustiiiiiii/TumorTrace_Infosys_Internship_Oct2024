# TumorTrace
Objective:
TumorTrace is a project aimed at leveraging deep learning techniques to classify tumor images into two categories: benign and malignant. By using state-of-the-art convolutional neural network (CNN) architectures, the project focuses on assisting in early and accurate detection of tumors, potentially aiding healthcare professionals in diagnosis.

# Dataset
The dataset consists of labeled images of tumors categorized into benign and malignant classes.
Pre-processing techniques are applied to enhance image quality and uniformity for training the model.

#Data Collection and Preprocessing
The dataset consists of MRI images labeled as malignant or benign. The data was collected from [source/database], ensuring a diverse representation of cases.

The preprocessing steps included:

Converting images to grayscale.

Resizing images to a standard dimension (e.g., 224x224).

Normalizing pixel values to a range suitable for neural

network input.

Augmenting the dataset using techniques such as rotation and
flipping.

 Model

VGG16: A pre-trained CNN model is fine-tuned for the classification task.
The use of transfer learning ensures efficient training even with limited data while leveraging the model's pre-learned features.

ResNet18: A lightweight model suitable for faster training with lower computational requirements. Ideal for initial benchmarks.

ResNet50: A deeper architecture that can capture more complex features, potentially increasing accuracy on tumor classification.


# Implementation Plan

Pre-trained ResNet models will be fine-tuned on the tumor dataset using transfer learning.


Comparative analysis will be conducted between VGG16, ResNet18, and ResNet50 based on accuracy, ROC-AUC, and inference time.


Evaluation Metrics
We evaluated our models using the following metrics:

Accuracy: Percentage of correct predictions.
Precision: Ability to avoid false positives.
Recall (Sensitivity): Ability to identify actual positives.
F1 Score: Harmonic mean of precision and recall.
AUC (Area Under Curve): Measures the model's ability to distinguish between classes.

Results and Impact
Model Performance:
VGG16:
Accuracy: 80.40%
AUC: 0.8518
ResNet18:
Accuracy: 77.945%
AUC: 0.8591
ResNet50:
Accuracy: 78.85%
AUC: 0.8471

LICENSE
This project is licensed under the MIT License - see the LICENSE file for details.

