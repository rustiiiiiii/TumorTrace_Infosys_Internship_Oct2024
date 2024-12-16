# TumorTrace
Project Tumor Trace: MRI-Based AI for Breast Cancer Detection

🔍 Project Overview

The Tumor Trace project focuses on creating an advanced AI model capable of detecting breast cancer through MRI images. Leveraging deep learning techniques, the goal is to develop precise predictions that assist radiologists in making early and accurate diagnoses.

📑 Table of Contents

Installation

Usage

Data Collection and Preprocessing

Model Architecture

Training

Evaluation Metrics

Results and Impact

Future Work

License

⚙️ Installation

To set up and run the project, you need to install the required libraries. Use the following command:

pip install -r requirements.txt

🚀 Usage

Steps to Use the Project:

Clone the repository:

git clone https://github.com/rustiiiiiii/TumorTrace_Infosys_Internship_Oct2024/tree/DARSHAN_MU
cd TumorTrace_Infosys_Internship_Oct2024

Prepare your MRI dataset:
Place the dataset in the directory specified in the project instructions.

Run the Jupyter Notebook:

jupyter notebook

Follow the instructions in the notebook:

Preprocess the data.

Train the model.

Evaluate its performance.

📊 Data Collection and Preprocessing

Data Source:

The dataset is sourced from Computers in Biology and Medicine and contains MRI images labeled as either malignant or benign. This ensures a diverse range of cases for robust training.

Preprocessing Steps:

Grayscale Conversion: All images were converted to grayscale for uniformity.

Resizing: Images were resized to a standard dimension of 224x224 pixels.

Normalization: Pixel values were normalized to a range suitable for neural network input.

Augmentation: To enhance the dataset, augmentation techniques such as rotation and flipping were applied.

Class Weights: Class weights were utilized to address imbalances in the dataset.

🏗️ Model Architecture

Overview:

The project utilizes Convolutional Neural Network (CNN) architectures, which are highly effective for extracting features from MRI images. Various models were tested:

VGG16:

Pre-trained on ImageNet.

Fine-tuned for binary classification (benign vs. malignant).

ResNet18 and ResNet50:

Deeper architectures.

Utilizes residual learning to improve training on complex datasets.

🏋️ Training

Training Configuration:

Best Accuracy: best_accuracy = 0

Epochs: total_epochs = 50

Momentum: momentum = 0.9

No CUDA Flag: no_cuda = False

Log Interval: log_interval = 10

Optimizer and Scheduler:

optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

📏 Evaluation Metrics

The following metrics were used to evaluate model performance:

Accuracy:

Measures the percentage of correct predictions.

Precision:

Evaluates the ability to avoid false positives.

Recall (Sensitivity):

Measures the ability to identify actual positives.

F1 Score:

The harmonic mean of precision and recall.

AUC (Area Under Curve):

Quantifies the model's ability to distinguish between classes.

Confusion Matrix:

Provides detailed performance in terms of true positives, false positives, true negatives, and false negatives.

📈 Results and Impact

Model Performance:

VGG16:

Accuracy: 81.70%

AUC: 0.8772

ResNet18:

Accuracy: 80.91%

AUC: 0.8859

ResNet50:

Accuracy: 80.43%

AUC: 0.8742

Impact:

The models demonstrated reliable accuracy and sensitivity, providing a strong foundation for aiding radiologists in detecting breast cancer.

🔮 Future Work

To enhance the project, the following improvements are planned:

Data Augmentation and Enrichment:

Incorporating larger and more diverse datasets.

Model Optimization:

Experimenting with advanced neural network architectures.

Integration with Medical Systems:

Developing APIs for seamless integration with existing diagnostic systems.

User Interface Improvements:

Enhancing the usability and accessibility of the project.

Real-time Inference and Deployment:

Optimizing for real-time inference and deploying on cloud platforms.

Explainability and Transparency:

Implementing techniques to improve the interpretability of model predictions.

📜 License

This project is licensed under the MIT License. For more details, refer to the LICENSE file in the repository.

Thank you for exploring Tumor Trace! Together, let's advance the fight against breast cancer using AI technology