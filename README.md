# Tumor Trace: AI-Based Breast Cancer Detection  

## Project Overview  
The Tumor Trace project leverages deep learning to detect breast cancer from MRI images. The objective is to develop a reliable model that classifies tumors as either malignant or benign, aiding radiologists in accurate and early diagnosis. 

---

## Table of Contents  
1. Data Collection and Preprocessing  
2. Model Architecture  
3. Training  
4. Evaluation Metrics  
5. Results  
6. Future Work  
7. License  

---

## Data Collection and Preprocessing  

### Data Overview  
The dataset consists of MRI images categorized into two classes: **malignant** (indicative of cancer) and **benign** (non-cancerous). It was sourced from [source/database], ensuring diverse patient demographics and imaging conditions.  

### Preprocessing Workflow  
The following steps were implemented to prepare the dataset:  
1. **Grayscale Conversion**: Simplified image representation by converting to a single intensity channel.  
2. **Resizing**: Standardized image dimensions to **224x224 pixels** for consistency across models.  
3. **Normalization**: Scaled pixel values to the range [0, 1], optimizing the model’s performance during training.  
4. **Data Augmentation**: Enhanced the dataset using techniques like **rotation** and **horizontal flipping** to mitigate overfitting.  

---

## Model Architecture  

The project employed convolutional neural networks (CNNs) for effective feature extraction from MRI images. Three architectures were analyzed:  

### **VGG16**  
- Utilized a pre-trained VGG16 model, modified for binary classification.  
- Fine-tuned to focus on distinguishing between benign and malignant tumors.  

### **ResNet18 and ResNet50**  
- Incorporated residual connections to mitigate vanishing gradient issues, enabling deeper network training.  
- These architectures provided robust performance in handling complex patterns within medical images.  

---

## Training  

### Training Setup  
The training process was configured as follows:  
- **Optimizer**: Adam optimizer for efficient weight updates.  
- **Loss Function**: Cross-Entropy Loss, with class weights to address dataset imbalances.  
- **Learning Rate Scheduler**: StepLR scheduler with a step size of 7 epochs and a gamma of 0.1.  
- **Epochs**: 50  
- **Batch Size**: 32  

Training was conducted on a GPU-enabled setup to manage the computational requirements effectively.  

---

## Evaluation Metrics  

The models were evaluated using the following metrics:  
- **Accuracy**: The proportion of correct predictions.  
- **Precision**: Focuses on minimizing false positive rates.  
- **Recall (Sensitivity)**: Measures the ability to correctly identify malignant cases.  
- **F1 Score**: Balances precision and recall for a comprehensive evaluation.  
- **AUC (Area Under the Receiver Operating Characteristic Curve)**: Reflects the model's discrimination capability between classes.  

---

## Results  

The performance of the evaluated models is summarized below:  

| **Model**  | **Accuracy** | **AUC**   |  
|------------|--------------|-----------|  
| VGG16      | 71.71%       | 0.5999    |  
| ResNet18   | 70.00%       | 0.5552    |  
| ResNet50   | 72.40%       | 0.7215    |  

---

## Future Work  

1. **Data Expansion**: Acquire a larger, more diverse dataset to improve model generalization.  
2. **Explainability**: Implement visualization techniques such as heatmaps to provide insights into model predictions.  
3. **Optimization**: Explore advanced architectures and optimize hyperparameters for enhanced performance.  
4. **Clinical Integration**: Design a seamless interface to enable real-world application in diagnostic workflows.  

---

## License  

This project is licensed under the **MIT License**. Refer to the [LICENSE](LICENSE) file for detailed terms.  
