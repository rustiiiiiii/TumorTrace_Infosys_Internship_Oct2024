# TumorTrace: Breast Cancer Detection Using MRI  

**TumorTrace** is an AI-powered project designed to detect breast cancer from MRI images. Leveraging advanced deep learning models, this project aids radiologists in early and reliable diagnosis of breast tumors.  

---

## Table of Contents  

- [Data Collection and Preprocessing](#data-collection-and-preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Results and Impact](#results-and-impact)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Data Collection and Preprocessing  

The dataset consists of MRI scans of breast tumors categorized as benign or malignant. It is divided into training, validation, and testing datasets:  

- **Training Set**:  
  - Benign: 5,559  
  - Malignant: 14,875  
- **Validation Set**:  
  - Benign: 408  
  - Malignant: 1,581  
- **Test Set**:  
  - Benign: 1,938  
  - Malignant: 4,913  

**Preprocessing steps include**:  
- Converting images to grayscale.  
- Resizing all images to 224x224 pixels.  
- Normalizing pixel values to match neural network input requirements.  
- Augmenting data with techniques like rotation and flipping to enhance model robustness.  

---

## Model Architecture  

This project employs **Convolutional Neural Networks (CNNs)** for effective feature extraction and classification. The following architectures were explored:  

- **VGG16**: A pre-trained model fine-tuned for binary classification (benign vs. malignant).  
- **ResNet18 and ResNet50**: Deeper architectures with residual learning for handling complex datasets and preventing vanishing gradient issues.  

---

## Training  

**Training Configuration**:  
- **Optimizer**: Adam  
- **Loss Function**: Binary Cross-Entropy Loss  
- **Learning Rate Scheduler**: StepLR (Step size = 7, Gamma = 0.1)  
- **Epochs**: 50  
- **Batch Size**: 32  

---

## Evaluation Metrics  

The models were evaluated using the following metrics:  
- **Accuracy**: Percentage of correct predictions.  
- **Precision**: Ability to minimize false positives.  
- **Recall (Sensitivity)**: Ability to identify true positives.  
- **F1-Score**: Harmonic mean of precision and recall.  
- **AUC (Area Under Curve)**: Measures the ability to distinguish between classes.  

---

## Results and Impact  

| **Model**  | **Accuracy** | **AUC**   | **Comments**               |  
|------------|--------------|-----------|----------------------------|  
| **VGG16**  | 74.27%       | 0.8226    | Moderate performance       |  
| **ResNet18** | 74.34%       | 0.8345    | Improved over VGG16        |  
| **ResNet50** | 77.54%       | 0.8549    | Best-performing model      |  

**Conclusion**:  
The **ResNet50** model achieved the best performance, making it the most reliable for deployment in real-world diagnostic scenarios.  

---

## Future Work  

1. Integrate multi-modal imaging (e.g., mammograms, ultrasounds) to improve classification accuracy.  
2. Explore advanced deep learning architectures to address false positives and false negatives.  
3. Deploy the model on cloud platforms for real-time clinical diagnostics.  

---

## License  

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

---
