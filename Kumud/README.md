# **TumorTrace: Breast Tumor Classification Using AI**

**TumorTrace** is an AI-powered system developed to classify breast tumors as **benign** or **malignant** using MRI images. The project leverages advanced deep learning models such as **ResNet18**, **ResNet50**, and **VGG16**, along with transfer learning techniques, to enhance accuracy and efficiency. By automating the tumor classification process, **TumorTrace** assists healthcare professionals in making faster, more accurate diagnoses, enabling early detection and better treatment outcomes.

### **Objective**

- **Automate Tumor Classification**: Classify breast tumors as benign or malignant using AI.
- **Early Detection**: Help healthcare professionals detect breast cancer at an earlier stage.
- **Improve Diagnostic Efficiency**: Speed up and enhance the accuracy of tumor classification.

---

## **Dataset**

The dataset for this project consists of medical images (MRI scans) in their corresponding folders (benign and malignant) for tumor detection. 


### Directory Structure
```text
classification_roi_path/
├── test/                 
│   ├── benign/            
│   └── malignant/         
├── train/                 
│   ├── benign/            
│   └── malignant/        
└── val/                   
    ├── benign/            
    └── malignant/         

```
---

## **Models Used**

1. **VGG16**: A 16-layer Convolutional Neural Network (CNN) known for its simplicity and effectiveness in image classification tasks. It is used as a baseline model for tumor classification.
  
2. **ResNet18**: A lighter version of the ResNet architecture with 18 layers, utilizing residual connections to allow deeper networks without the vanishing gradient problem. Ideal for efficient training and smaller datasets.

3. **ResNet50**: A deeper ResNet model with 50 layers, designed to capture complex patterns in the data, offering high accuracy for classifying tumors in more complex datasets.

---

## **Installation**

### **Prerequisites**
1. Python 3.8 or later
2. Required libraries: TensorFlow, PyTorch, Gradio, scikit-learn , numpy , pandas , matplotlib , seaborn , Pillow 

### **Steps**

1. Create a Kaggle Notebook
    - Go to Kaggle and log in.
    - Create a new notebook by selecting "New Notebook" from the "Code" section.
      
2. Set Up Your Kaggle Environment
   - Install required libraries (if not already pre-installed) in the Kaggle notebook

3. Upload Dataset to Kaggle
    In the "Data" section of the Kaggle notebook, click on "Add Data" and upload your dataset.
    ```bash
    dataset_path = '/kaggle/input/your-dataset-name/'

4. Train the Model - **tumortrace.ipynb**
   - Clone the repository:
      ```bash
      git clone https://github.com/rustiiiiiii/TumorTrace_Infosys_Internship_Oct2024/tree/Kumud/Kumud/Code/tumortrace.ipynb
      
   - Run Cells Sequentially - 
     To execute cells sequentially, click the "Run All" button , or run each cell individually.
     
   - Once the training is complete, the model will be saved as a file, typically in the models/ directory. By default, the saved model will be       named vgg16.pth , resnet18.pth , resnet50.pth .

5.  Run Gradio Interface for Predictions - **tumortrace-application.ipynb**
  - Clone the repository:
      ```bash
      git clone https://github.com/rustiiiiiii/TumorTrace_Infosys_Internship_Oct2024/tree/Kumud/Kumud/Code/tumortrace-application.ipynb
      
   - Run Cells Sequentially - 
     To execute cells sequentially, click the "Run All" button , or run each cell individually.
     
6. Access the Web Interface
   
---

## **Results**

We evaluated three different models for the tumor classification task: **Vgg16**, **Resnet18**, and **Resnet50**. Each model was trained and tested on the same dataset to compare their performance in distinguishing between **Benign** and **Malignant** tumor images.

### Model Performance

| Model         | Accuracy | 
|---------------|----------|
| **Vgg16**     | 78.69%   | 
| **Resnet18**  | 76.66%   | 
| **Resnet50**  | 75.61%   | 

### Gradio Interface
![image](https://github.com/user-attachments/assets/0dfbf718-9e1d-41fe-b8b6-31cf61074334)

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.

## **Acknowledgements**
    Infosys Springboard 5.0: For providing the platform and resources for this project.
    Mentor - Anurag Sista: For invaluable guidance and support throughout the development process.
