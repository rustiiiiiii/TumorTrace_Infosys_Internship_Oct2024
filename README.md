# TumorTrace


# MRI-Based Breast Cancer Detection

## Overview
This project focuses on leveraging Artificial Intelligence (AI) and Machine Learning (ML) techniques to detect breast cancer from MRI scans. The system is designed to differentiate between benign and malignant lesions, providing a step toward early detection and diagnosis of breast cancer. By integrating advanced feature extraction techniques and user-friendly tools, this project aims to aid healthcare professionals in decision-making.

---

## Key Features
### 1. **Preprocessing**
- **Data Augmentation**: Implements techniques such as rotation, flipping, and scaling to increase dataset diversity and improve model generalization.
- **Normalization**: Standardizes MRI image pixel values to enhance consistency.

### 2. **Feature Extraction**
- **HOG**: Utilized to capture gradient orientation and magnitude information, highlighting textures and shapes in MRI images.
- **Sobel Edge detection**: Applied to detect edges by computing the gradient of pixel intensity in the horizontal and vertical directions, emphasizing boundaries in MRI scans.
- **Local Binary Patterns (LBP)**: Utilized to extract texture features from MRI images.
- **Mean-Variance-Median Local Binary Patterns (MVM-LBP)**: Advanced descriptor for texture analysis based on the Mean, Variance, and Median values of pixel intensity.

### 3. **Model Training**
- Uses pre-trained models like VGG-16, ResNet-18 and ResNet-50
- Emphasizes model evaluation metrics such as accuracy, precision, recall, specificity, displays Confusion Matrix, ROC curve, AUC

### 4. **Interactive Interface**
- **Gradio Integration**: Provides an interactive web interface to test the models with sample MRI images, making it accessible to non-technical users.

---

## Technologies Used
- **Python Libraries**:
  - PyTorch (for model development and training)
  - NumPy and pandas (for data manipulation)
  - Scikit-learn (for ML model evaluation)
  - OpenCV (for image processing)
- **Gradio**: For building a web-based demonstration interface.
- **Google Colab**: Used for model development and experimentation.
- **Kaggle**: Utilized for running experiments.

---

## Project Structure
```
├── code/
│   ├── gradiocode.ipynb         # Notebook for Gradio interface
│   ├── infosys-mri-2 (2).ipynb  # Main notebook for preprocessing, training, and testing
├── DOCUMENTATION/
│   ├── PROJECT REPORT_merged.pdf # merged project report submissions
├── README.md                   # Project documentation
├── LICENSE                     # Licensing information
```

---

## Getting Started

### 1. Clone the Repository

### 2. Set Up Dependencies
Install the required Python packages:

### 3. Run the Notebooks
- Open the notebook `infosys-mri-2 (2).ipynb` to preprocess data, extract features, and train the models.
- Use the `gradiocode.ipynb` notebook to launch the Gradio interface.

---

## How to Use
1. Upload an MRI scan image through the Gradio interface.
2. The system will process the image, extract features, and classify the lesion as either **Benign** or **Malignant**.
3. The results, along with confidence scores, will be displayed.

---

## Future Enhancements
- Incorporate other deep learning techniques for improved accuracy.
- Experiment with other models.
- Extend the system to handle multi-class classification for various lesion types.
- Integrate deployment-ready APIs for clinical use.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributing
Contributions are welcome! Feel free to raise issues, suggest improvements, or submit pull requests.

---

## Acknowledgements
Special thanks to the Infosys Springboard program for providing resources and mentorship throughout this project. The project is inspired by the need for AI-driven advancements in healthcare.

---
