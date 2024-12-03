
# MRI Breast Tumor Classifier

This project is an MRI-based breast tumor classification application that predicts the likelihood of a tumor being malignant or benign. The application is deployed using **Gradio** for a user-friendly interface. The classification is performed using three pre-trained deep learning models: **ResNet18**, **ResNet50**, and **VGG16**.

---

## Features
- **Input:** MRI images of the breast.
- **Output:** Classification results (Malignant or Benign) with prediction confidence.
- **Models Used:**
  - ResNet18 (Accuracy: 68%)
  - ResNet50 (Accuracy: 76%)
  - VGG16 (Accuracy: 79%)
- **Deployment:** User-friendly web interface built using Gradio.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or later
- Required libraries: TensorFlow, PyTorch, Gradio, scikit-learn , numpy , pandas , matplot, jupyter notebook

### Steps
1. Create a python enviornmet:
  ```bash
    python -m venv venv
  ```
2. Activate python enviornment
  ```bash
      .venv\Scripts\activate
  ```
   
3 . Install dependencies mentioned above using pip install liberary name
  ```bash
      pip install <liberary-name>
  ```
3. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mri-breast-tumor-classifier.git
   cd mri-breast-tumor-classifier
   ```
4 Run jupyter notebook:
  ```bash
    jupyter notebook
  ```
5 In jupyter notebook open notebook file in Code folder and run

## Models and Training
- **ResNet18:** A lightweight model achieving an accuracy of 68%.
- **ResNet50:** Deeper architecture with better performance, achieving 76% accuracy.
- **VGG16:** Achieves the highest accuracy of 79%, making it the most reliable among the three models.

Each model was fine-tuned on the dataset for optimal performance.

---

## Results
| Model    | Accuracy |
|----------|----------|
| ResNet18 | 68%      |
| ResNet50 | 76%      |
| VGG16    | 79%      |

---

## Usage
1. Upload an MRI image using the Gradio interface.
2. Select the model (ResNet18, ResNet50, or VGG16) to use for classification.
3. View the classification result and confidence score.

---

## Future Improvements
- Implement additional deep learning models to further improve accuracy.
- Enhance the dataset by adding more diverse samples.
- Deploy the application on cloud platforms for scalability.

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes or feature enhancements.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **Frameworks:** PyTorch, TensorFlow, Gradio
- **Model Architectures:** ResNet, VGG
- **Dataset:** [Add dataset source or reference here]
