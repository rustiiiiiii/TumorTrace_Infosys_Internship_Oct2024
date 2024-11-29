# TumorTrace
<hr>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#features">Features</a></li>
    <li><a href="#technologies">Technologies Used</a></li>
    <li><a href="#structure">Project Structure</a></li>
    <li><a href="#setup">Setup and Installation</a></li>
    <li><a href="#dataset">Dataset</a></li>
    <li><a href="#model">Model Description</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#gradio">Gradio Interface</a></li>
    <li><a href="#license">License</a></li>
</ul>

<hr>

<h2 id="features">Features</h2>
<ul>
    <li><strong>Image Classification</strong>: Classifies breast cancer MRI scans as <strong>Benign</strong> or <strong>Malignant</strong>.</li>
    <li><strong>Deep Learning Model</strong>: Utilizes ResNet18, a convolutional neural network (CNN), fine-tuned on a breast cancer dataset.</li>
    <li><strong>Data Preprocessing</strong>: Includes image resizing, augmentation, normalization, and dataset splitting for training and testing.</li>
    <li><strong>Interactive Gradio Interface</strong>: Offers a user-friendly web interface for uploading images and getting real-time predictions.</li>
</ul>

<hr>

<h2 id="technologies">Technologies Used</h2>
<ul>
    <li><strong>Python</strong>: The primary programming language.</li>
    <li><strong>PyTorch</strong>: Deep learning framework used to develop and fine-tune the model.</li>
    <li><strong>Gradio</strong>: For creating an interactive web-based user interface.</li>
    <li><strong>OpenCV</strong>: For image preprocessing tasks.</li>
    <li><strong>Scikit-learn</strong>: For performance metrics and model evaluation.</li>
    <li><strong>Matplotlib</strong>: For visualizing data, training progress, and results.</li>
</ul>

<hr>

<h2 id="structure">Project Structure</h2>
<hr>

<h2 id="setup">Setup and Installation</h2>

<h3>Prerequisites</h3>
<p>Ensure the following tools and libraries are installed:</p>
<ul>
    <li>Python 3.6 or higher</li>
    <li>CUDA (if using GPU for training)</li>
</ul>

<h3>Installation Steps</h3>
<ol>
    <li>Clone the repository:
        <pre>
        git clone https://github.com/rustiiiiiii/TumorTrace_Infosys_Internship_Oct2024.git
        cd TumorTrace_Infosys_Internship_Oct2024
        </pre>
    </li>
    <li>Install the required dependencies:
        <pre>
        pip install -r requirements.txt
        </pre>
    </li>
    <li>Ensure the dataset is placed in the <code>data/</code> directory (see <a href="#dataset">Dataset</a>).</li>
</ol>

<hr>

<h2 id="dataset">Dataset</h2>
<p>The model is trained on MRI images of breast cancer tumors, labeled as <strong>Benign</strong> or <strong>Malignant</strong>.</p>
<p>A suitable dataset can be downloaded from <a href="https://www.kaggle.com/datasets">Kaggle Breast Cancer Datasets</a>.</p>
<p>Dataset preprocessing includes resizing, normalization, and augmentation to enhance model performance.</p>

<hr>

<h2 id="model">Model Description</h2>

<h3>ResNet18 Architecture</h3>
<ul>
    <li><strong>ResNet18</strong> is a convolutional neural network pre-trained on ImageNet.</li>
    <li>Transfer learning is applied by fine-tuning the model on the breast cancer dataset.</li>
    <li>Key steps in model development:
        <ol>
            <li>Preprocess the dataset.</li>
            <li>Train the model on labeled MRI images.</li>
            <li>Save the trained model in the <code>models/</code> directory for inference.</li>
        </ol>
    </li>
</ul>

<hr>

<h2 id="usage">Usage</h2>

<h3>Training the Model</h3>
<pre>
python train.py
</pre>

<h3>Testing the Model</h3>
<pre>
python test.py
</pre>

<hr>

<h2 id="gradio">Gradio Interface</h2>
<p>The project includes a Gradio app for real-time predictions.</p>
<ol>
    <li>Launch the Gradio app:
        <pre>
        python app.py
        </pre>
    </li>
    <li>Open the provided URL (usually <code>http://localhost:7860</code>) in your web browser.</li>
    <li>Upload an MRI image to get a prediction (either <strong>Benign</strong> or <strong>Malignant</strong>) along with a confidence score.</li>
</ol>

<hr>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>

<hr>


