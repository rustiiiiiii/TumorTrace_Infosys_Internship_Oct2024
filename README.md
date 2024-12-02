#TumorTrace: Breast Cancer Detection Using MRI
TumorTrace is an AI-based project aimed at detecting breast cancer using MRI images. By leveraging state-of-the-art deep learning models, the project seeks to provide accurate predictions, assisting radiologists in early and reliable diagnosis.
Data Collection and Preprocessing
The dataset consists of breast tumor MRI scans, divided into train, validation, and test sets:

Train: 5,559 Benign | 14,875 Malignant
Validation: 408 Benign | 1,581 Malignant
Test: 1,938 Benign | 4,913 Malignant
All images are resized to 224x224 pixels for compatibility with deep learning models

Preprocessing steps:

Converted images to grayscale.
Resized images to 224x224 pixels.
Normalized pixel values for neural network input.
Augmented data using rotation and flipping.
Model Architecture
The project employs Convolutional Neural Networks (CNNs) for feature extraction and classification. The following architectures were explored:

VGG16:
Fine-tuned pre-trained VGG16 model for binary classification.
ResNet18 and ResNet50:
Deeper architectures with residual learning for enhanced training on complex datasets.
Training
The models were trained using the following configuration:

Optimizer: Adam
Loss Function: Binary Cross-Entropy Loss
Learning Rate Scheduler: StepLR with a step size of 7 and gamma of 0.1
Epochs: 50
Batch Size: 32
Evaluation Metrics
We evaluated the models using the following metrics:

Accuracy: Percentage of correct predictions.
Precision: Ability to avoid false positives.
Recall (Sensitivity): Ability to identify actual positives.
F1-Score: Harmonic mean of precision and recall.
AUC (Area Under Curve): Measures the model's ability to distinguish between classes.

Results and Impact
Model	Accuracy	AUC	Comments
VGG16	74.27%	0.8226	Moderate performance
ResNet18	74.34%	0.8345	Improved over VGG16
ResNet50	77.54%	0.8549	Best-performing model
The ResNet50 model demonstrated the highest accuracy and AUC, suggesting it is the most reliable for real-world deployment.

Future Work
Integrate multi-modal imaging (e.g., mammograms and ultrasounds) to enhance classification accuracy.
Address false positives/negatives by exploring advanced architectures.
Deploy the model on the cloud for real-time diagnostics in clinical settings.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

