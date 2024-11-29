# TumorTrace
Objective:
TumorTrace is a project aimed at leveraging deep learning techniques to classify tumor images into two categories: benign and malignant. By using state-of-the-art convolutional neural network (CNN) architectures, the project focuses on assisting in early and accurate detection of tumors, potentially aiding healthcare professionals in diagnosis.

# Dataset
The dataset consists of labeled images of tumors categorized into benign and malignant classes.
Pre-processing techniques are applied to enhance image quality and uniformity for training the model.

 # Model

VGG16: A pre-trained CNN model is fine-tuned for the classification task.
The use of transfer learning ensures efficient training even with limited data while leveraging the model's pre-learned features.

ResNet18: A lightweight model suitable for faster training with lower computational requirements. Ideal for initial benchmarks.

ResNet50: A deeper architecture that can capture more complex features, potentially increasing accuracy on tumor classification.


# Implementation Plan

Pre-trained ResNet models will be fine-tuned on the tumor dataset using transfer learning.


Comparative analysis will be conducted between VGG16, ResNet18, and ResNet50 based on accuracy, ROC-AUC, and inference time.

# Outcome

Insights into the performance of different CNN architectures for medical imaging tasks.

Selection of the most efficient model for deployment in real-world applications.
