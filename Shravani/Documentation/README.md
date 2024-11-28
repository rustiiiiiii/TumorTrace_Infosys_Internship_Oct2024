##  Project Overview

  

The **Tumor Trace** project aims to develop an AI-based model for detecting breast cancer using MRI images. By leveraging deep learning techniques, the goal is to provide accurate predictions to assist radiologists in early diagnosis.

  

  

##  Table of Contents

  

-  [Installation](#installation) 

-  [Usage](#usage)  

-  [Data Collection and Preprocessing](#data-collection-and-preprocessing)

-  [Model Architecture](#model-architecture)

-  [Training](#training)

-   [Evaluation Metrics](#evaluation-metrics)

-   [Results and Impact](#results-and-impact)

-   [Future Work](#future-work)
  
  

  

##  Installation

  

To run this project, you need to install the necessary libraries. You can do this by executing the following command:

  

```shell

pip  install  -r  requirements.txt

```

  

  

##  Usage

1. Clone the repository

```shell

git  clone  https://github.com/yourusername/TumorTrace.git

  

cd  TumorTrace

```

  

2. Place your MRI dataset in the specified directory.

  

3. Run the Jupyter Notebook:

```shell

jupyter  notebook

```

  

4. Follow the instructions in the notebook to preprocess data, train the model, and evaluate its performance.

  

  

##  Data Collection and Preprocessing

  

  

The dataset consists of MRI images labeled as malignant or benign. The data was collected from [source/database], ensuring a diverse representation of cases.

  

  

The preprocessing steps included:

  

  

- Converting images to grayscale.

- Resizing images to a standard dimension (e.g., 224x224).

- Normalizing pixel values to a range suitable for neural

network input.

- Augmenting the dataset using techniques such as rotation and

flipping.

  

  

##  Model Architecture
The model uses a Convolutional Neural Network (CNN) architecture designed to extract features from MRI images effectively. We experimented with several models, including VGG16, ResNet18, and ResNet50.

### VGG16:

-   Pre-trained VGG16 model fine-tuned for binary classification (benign vs. malignant).

### ResNet18 and ResNet50:

-   Deeper architectures with residual learning for improved training on complex datasets.

## Training

We trained our models using the following setup:

-   **Optimizer**: Adam
-   **Loss Function**: Cross-Entropy Loss with class weights
-   **Learning Rate Scheduler**: StepLR with a step size of 7 and gamma of 0.1
-   **Epochs**: 50
-   **Batch Size**: 32

## Evaluation Metrics

We evaluated our models using the following metrics:

-   **Accuracy**: Percentage of correct predictions.
-   **Precision**: Ability to avoid false positives.
-   **Recall (Sensitivity)**: Ability to identify actual positives.
-   **F1 Score**: Harmonic mean of precision and recall.
-   **AUC (Area Under Curve)**: Measures the model's ability to distinguish between classes.

## Results and Impact

### Model Performance:

-   **VGG16**:
    -   Accuracy: 78.60%
    -   AUC: 0.8518
-   **ResNet18**:
    -   Accuracy: 77.93%
    -   AUC: 0.8591
-   **ResNet50**:
    -   Accuracy: 76.40%
    -   AUC: 0.8471