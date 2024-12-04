

# Breast Cancer Detection using ResNet18

This project leverages a ResNet18 model to classify breast cancer images as either **Benign** or **Malignant**. The model is trained on a breast cancer dataset, and a Gradio app is used to provide an interactive interface where users can upload images to get real-time predictions.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset](#dataset)
- [Model](#model)
- [Gradio Interface](#gradio-interface)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction
This repository contains a machine learning model for breast cancer detection. The model is built using **ResNet18**, a convolutional neural network (CNN) architecture pre-trained on ImageNet. The model is fine-tuned on a dataset of breast cancer images to classify whether the input is **Benign** or **Malignant**. 

To make it user-friendly, the project uses a Gradio app as the frontend, allowing users to upload images and view real-time predictions.

## Setup

### Requirements
- Python 3.6 or higher
- PyTorch 1.x or higher
- Gradio 2.x or higher
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

### Installation

Clone this repository to your local machine:

```bash
git clone rustiiiiiii/TumorTrace_Infosys_Internship_Oct2024
cd shalem
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The model is trained on a breast cancer dataset, such as the **Breast Cancer Histopathology Images Dataset** from Kaggle. The dataset contains images of breast cancer samples, labeled as **Benign** or **Malignant**. The images are resized and preprocessed to be compatible with the ResNet18 model.

- Download the dataset from: [Kaggle Breast Cancer Dataset](https://www.kaggle.com/datasets) (You may need to search for an appropriate dataset).
- Ensure the dataset is placed in the correct directory (as specified in the script).

## Model

The model used in this project is based on **ResNet18**, which is a deep residual network with 18 layers. This architecture is popular for image classification tasks, and we use transfer learning, where the model is first pre-trained on ImageNet and then fine-tuned on the breast cancer dataset.

### Model Training

1. Preprocess the dataset (resize, normalization, and augmentation).
2. Split the dataset into training and testing sets.
3. Fine-tune the pre-trained ResNet18 model on the breast cancer dataset.
4. Save the trained model weights for later use.

## Gradio Interface

A Gradio app is used to provide a simple and interactive web interface. The app allows users to upload an image of a breast cancer sample and immediately get a prediction.

To run the Gradio interface, execute the following command:

```bash
python app.py
```

This will launch a local server. Open a web browser and go to the provided URL (usually `http://localhost:7860`) to use the interface.

## How to Use

1. Run the `app.py` file:
   ```bash
   python app.py
   ```
2. A web interface will open where you can upload an image of a breast cancer sample.
3. Once the image is uploaded, the model will classify it as **Benign** or **Malignant** based on the prediction from the ResNet18 model.
4. The result will be displayed on the interface along with a confidence score.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the file further with additional details, such as how to run the training script or specific dataset instructions, depending on your project setup.
