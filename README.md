TumorTrace: MRI-Based AI for Breast Cancer Detection

Project Brief

TumorTrace is an AI-powered solution for detecting and classifying breast tumors as benign or malignant using MRI images. The project leverages deep learning models like VGG16, ResNet-50, and ResNet-18 for accurate tumor classification. The goal is to automate tumor detection, aiding in faster and more reliable diagnoses.

Key Features:

	

- Utilizes pre-trained models like VGG16, ResNet-50, and ResNet-18 for feature extraction and fine-tuning.
- Advanced image processing techniques like Local Binary Patterns (LBP), Gray Level Co-occurrence Matrix (GLCM), and Sobel Edge Detection for better feature extraction.
- Real-time predictions using a Gradio-based interface for user interaction.

Dependencies

1. Install Dependencies

To install all the required libraries for this project, run the following command:

```bash
pip install torch torchvision gradio numpy matplotlib tqdm scikit-learn scikit-image opencv-python
```

2. Libraries Breakdown:

- **torch**: Core library for building deep learning models.
- **torchvision**: Provides image transformations and pre-trained models.
- **gradio**: For creating real-time prediction interfaces.
- **numpy**: For numerical operations and array manipulation.
- **matplotlib**: For visualizations such as confusion matrices and ROC curves.
- **tqdm**: Progress bars for loops, like during model training.
- **scikit-learn**: For classification metrics like accuracy and confusion matrix.
- **scikit-image**: For image processing tasks like LBP and GLCM.
- **opencv-python**: For additional image processing, including Sobel edge detection.

2. Data Loading and Preprocessing

 After loading the data, the following preprocessing steps should be applied:
 
   2.1. **Resizing**:
   - Resize all images to **224x224 pixels** to match the input size required by the models.

   2.2. **Normalization**:
   - Normalize the pixel values using **ImageNet**’s mean and standard deviation values to ensure consistent inputs for the model.

   2.3. **Data Augmentation**:
   - **Random Horizontal and Vertical Flips**: Applied with a 50% chance.
   - **Random Rotation**: Images are rotated within a range of **-20 to 20 degrees**.
   - **Color Jitter**: Adjusts brightness, contrast, and saturation to mimic real-world variations.

   2.4. **Feature Extraction**:
   - **Sobel Edge Detection**: Highlights edges in the images for better feature representation.
   - **Local Binary Patterns (LBP)**: Extracts texture features for improved classification.
   - **Gray Level Co-occurrence Matrix (GLCM)**: Analyzes spatial relationships between pixels to extract texture-based features.
3. Run the Training

Training the model involves setting up the training loop, defining the loss function, and specifying the SGD optimizer.

```python
for epoch in range(50):
    train(epoch, model, 50, train_loader, criterion, 0.01, 0.01)
```
4. Test the Model

After training the model, it’s important to evaluate it on a test or validation set to assess its performance. 
```python
test(model, test_loader)
```

5.Outputs

#### Model Accuracy:
The evaluation results for the models on the test set are:

- **VGG16**: 75.52%
- **ResNet-50**: 73.07%
- **ResNet-18**: 72.62%

#### Classification Metrics:
- **Accuracy**: The percentage of correct predictions.
- **Sensitivity (Recall)**: The model’s ability to detect malignant cases.
- **Specificity**: The model’s ability to detect benign cases.
- **AUC (Area Under the Curve)**: Represents how well the model distinguishes between benign and malignant cases.
- **Precision**: The proportion of true positives among all positive predictions.
- **F1-Score**: The harmonic mean of precision and recall.

#### Visualizations:
- **Confusion Matrix**: Shows the number of correct and incorrect predictions for each class (Benign and Malignant).
- **ROC Curve**: Plots the trade-off between sensitivity and specificity across different thresholds.

6. Real-Time Predictions with Gradio

   To make real-time predictions after training the model, used Gradio to create a simple interface where users can upload images and select a model for classification.




