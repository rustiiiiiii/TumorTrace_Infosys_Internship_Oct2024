Tumor Trace Project
An AI-based solution to assist radiologists in detecting breast cancer using MRI images. Leveraging deep learning techniques, this project aims to provide accurate early diagnosis support.

Table of Contents
Installation
Usage
Data Collection and Preprocessing
Model Architecture
Training
Evaluation Metrics
Results and Impact
Future Work
Installation
To set up the project, follow these steps:

Install the required libraries:
pip install -r requirements.txt  
Usage
Clone this repository:

git clone https://github.com/yourusername/TumorTrace.git  
cd TumorTrace  
Place your MRI dataset in the designated folder.

Launch the Jupyter Notebook:

jupyter notebook  
Follow the instructions in the notebook for data preprocessing, training, and model evaluation.

Data Collection and Preprocessing
Dataset: MRI images labeled as malignant or benign, sourced from a [specified database].

Preprocessing Steps:

Convert images to grayscale.
Resize images to 224x224 pixels.
Normalize pixel values.
Augment data (e.g., rotations, flipping).
Model Architecture
This project utilizes Convolutional Neural Networks (CNNs) for effective feature extraction from MRI images.
We experimented with the following models:

1. VGG16
Pre-trained on ImageNet and fine-tuned for binary classification.
2. ResNet18 & ResNet50
Deeper architectures with residual connections to handle complex datasets.
Training Setup
Optimizer: Adam
Loss Function: Cross-Entropy with class weights
Learning Rate Scheduler: StepLR (step size: 7, gamma: 0.1)
Epochs: 50
Batch Size: 32
Evaluation Metrics
We assessed the models using:

Accuracy: Correct predictions as a percentage of total samples.
Precision: Ability to avoid false positives.
Recall (Sensitivity): Correct identification of actual positives.
F1 Score: Balance between precision and recall.
AUC (Area Under the Curve): Ability to distinguish between classes.
Results and Impact
Model Performance
Model	Accuracy	AUC
VGG16	78.60%	0.8518
ResNet18	77.93%	0.8591
ResNet50	76.40%	0.8471
Impact: The results demonstrate the potential of deep learning to improve breast cancer detection accuracy, supporting radiologists in early diagnosis.