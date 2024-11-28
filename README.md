# TumorTrace

	1.	Objective:
The goal is to create a system that can automatically analyze MRI images of the breast to detect tumors and classify them as benign (non-cancerous) or malignant (cancerous). This is done by leveraging deep learning algorithms that can learn from labeled MRI data and improve detection accuracy over time.
	2.	Data Collection:
The dataset used for the project consists of MRI images of breast tumors, which are labeled into two classes: benign and malignant. The data is divided into training, validation, and test sets to ensure that the models are properly trained and evaluated.
	3.	Preprocessing:
The project involves several preprocessing steps to prepare the MRI images for training the models. These include resizing images, normalizing pixel values, data augmentation (such as flipping and rotating images), and feature extraction (e.g., using Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP)).
	4.	Model Training:
The project uses deep learning models like VGG16, ResNet50, and ResNet18. These models are pre-trained on large datasets (like ImageNet) and then fine-tuned for breast tumor detection using the MRI images. The models are trained using techniques like CrossEntropyLoss for binary classification, and optimizers like Adam and SGD are used to minimize the loss during training.
	5.	Model Evaluation:
After training, the models are evaluated on a separate test dataset
	6.	Results:
The results show that the VGG16 model achieves the highest accuracy at 75.52%, followed by ResNet50 at 73.07%, and ResNet18 at 72.62%. These models show promising results in accurately classifying breast tumors from MRI scans.
	7.	Real-Time Application:
The project also includes the development of an interactive web-based application using Gradio. In this application, users can upload MRI images and select a model (VGG16, ResNet50, or ResNet18). The model then classifies the tumor as benign or malignant and displays the probabilities for each class.
