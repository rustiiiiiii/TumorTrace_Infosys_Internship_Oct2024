Certainly! Here's the updated version of the `README.md` file, with the code snippet replaced by the relevant file path:

```markdown
# Breast Cancer Tumor Detection Using ResNet18 and Gradio

This project is designed for detecting breast cancer tumors (either **Benign** or **Malignant**) using the **ResNet18** model. The model is trained on image data of breast cancer histopathological slides to classify tumors based on input images. The application leverages **Gradio** for an interactive web interface to classify new images.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Gradio Interface](#gradio-interface)
- [Conclusion](#conclusion)

## Project Overview

This project uses a **ResNet18** deep learning model to classify breast cancer tumors as either **Benign** or **Malignant**. It automates the diagnostic process and provides a Gradio-based user interface where users can upload an image and get real-time predictions.

The Jupyter Notebook (`Gradio.ipynb`) contains the following steps:
1. Model definition and loading.
2. Preprocessing of input images.
3. Gradio interface setup for image classification.
4. Prediction of new images through the Gradio app.

## Requirements

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Gradio
- TorchVision
- Pillow

You can install the necessary packages using `pip`:

```bash
pip install gradio torch torchvision pillow
```

## Installation Instructions

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone https://github.com/yourusername/breast-cancer-tumor-detection.git
   ```

2. **Navigate to the project folder**:

   ```bash
   cd breast-cancer-tumor-detection
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:

   To open the notebook with Gradio interface:

   ```bash
   jupyter notebook Gradio.ipynb
   ```

## Usage

Once the notebook is open, follow the instructions in the notebook to:
1. Load the pretrained **ResNet18** model.
2. Set up the Gradio interface to classify new images.
3. Upload an image of a breast cancer slide (either **Benign** or **Malignant**) and receive a prediction.

### Example:

To use the Gradio interface:
1. Run the notebook to launch the Gradio app.
2. In the app, upload a new image for prediction.
3. The app will return a classification of the image as either **Benign** or **Malignant**, along with the confidence scores.

## Model Details

- **ResNet18**: A convolutional neural network architecture with 18 layers, pre-trained on ImageNet.
- The model is fine-tuned for binary classification to classify breast cancer images.

### Architecture:
- The **ResNet18** model is used with transfer learning. The last fully connected layer is replaced with a custom layer for binary classification (Benign/Malignant).
  
### Input:
- The input to the model is an image of a breast tissue slide, usually in `.jpg` or `.png` format.

### Output:
- The model outputs a label: **Benign** or **Malignant** based on the image input.

## Training Process

The training process involves:
1. Loading the dataset of breast cancer images.
2. Preprocessing the images (resize, normalization).
3. Fine-tuning the **ResNet18** model for classification.
4. Training the model with a binary cross-entropy loss function.

During training, the model’s performance is monitored through accuracy and loss metrics.

## Evaluation

After training, the model is evaluated using:
- **Accuracy**: Performance on the test dataset.
- **Confusion Matrix**: Visualizing true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Includes Precision, Recall, and F1-Score.

## Gradio Interface

The Gradio interface allows users to interact with the model and classify breast cancer images through a simple web app.

- **Image Input**: Users can upload an image of a breast tissue slide.
- **Output**: The model predicts whether the tumor is **Benign** or **Malignant**.
- **Confidence Scores**: The app also provides confidence scores for each prediction.

### File Path for Gradio Interface Code

The code for the Gradio interface is located in the Jupyter notebook file:

```
Shalem/Code/Gradio.ipynb
```

This notebook contains the following:
1. **Model loading**: The pretrained ResNet18 model is loaded from a specified path.
2. **Image preprocessing**: Images are resized and normalized for input into the model.
3. **Gradio interface**: The Gradio interface is set up to allow image uploads and predictions.

### Running the Interface:

The Gradio app will launch automatically when you run the notebook, allowing you to interact with the model directly in your browser.

## Conclusion

This project demonstrates how to use **ResNet18** for classifying breast cancer images and how to integrate **Gradio** for a user-friendly interactive interface. By using this approach, we can automate the diagnosis process, enabling quicker and more accurate predictions for better patient outcomes.

---

Make sure to adjust paths and file names based on your actual project structure. Let me know if you need further updates!
```

In this version, the **code snippet** for the Gradio interface has been replaced with the path to the file that contains it:

```
Shalem/Code/Gradio.ipynb
```

Let me know if you need further modifications or additions!
