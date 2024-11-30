TumorTrace: MRI-Based AI for Breast Cancer Detection

TumorTrace is a deep learning-based project designed to classify breast cancer from MRI images using advanced CNN architectures, such as VGG16, ResNet18, and ResNet50. The system helps in the early detection of breast cancer by predicting whether a tumor is benign or malignant, providing valuable assistance to healthcare professionals.
Key Features

    Deep Learning Models: Utilizes VGG16, ResNet18, and ResNet50 for classifying tumors.
    Real-Time Prediction: Interactive user interface built with Gradio for real-time tumor classification.
    Model Evaluation: Evaluates performance using accuracy, loss, AUC, precision, recall, and F1-score.

The project implements the following models:

    VGG16: Known for its depth and simplicity, suitable for extracting features from MRI images.
    ResNet18: Uses residual connections to improve training of deeper networks.
    ResNet50: A deeper variant of ResNet offering high accuracy for complex image classification tasks.

Dataset(https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205)

The model is trained and tested on a dataset of breast cancer MRI images, which is preprocessed to ensure consistency and proper normalization. The dataset is divided into training, validation, and test sets for robust evaluation.
Evaluation Metrics

Gradio Interface

An interactive Gradio interface allows users to upload MRI images for real-time predictions, displaying whether the tumor is benign or malignant.

License

This project is licensed under the MIT License.
