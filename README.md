# ♻️E-Waste Generation Classification using EfficientNetV2B0 (Transfer Learning)

This project was developed as part of the Edunet Foundation AI/ML Virtual Internship, with the goal of building a deep learning model that classifies different types of electronic waste (e-waste) into predefined categories using image data and transfer learning.

---

## 📌 Problem Statement

E-waste (electronic waste) is rapidly becoming a serious environmental and health issue around the world. Proper sorting and categorization of e-waste is essential for efficient recycling and disposal, but manual classification is error-prone and labor-intensive.

This project aims to automate e-waste classification using deep learning techniques, allowing for accurate image-based classification of e-waste types.

---

## 🎯 Objective

To build a deep learning image classifier using EfficientNetV2B0 (via transfer learning) that can classify images of e-waste into 10 categories with high accuracy and deploy it using Gradio as an interactive web app.

---

## 🧠 Technical Stack

* Python
* TensorFlow / Keras
* EfficientNetV2B0
* NumPy, Matplotlib, Seaborn
* Scikit-learn
* Gradio (for deployment)
* Google Colab (for training)

---

## 🗂️ Dataset

* **Source:** [Kaggle - E-Waste Image Dataset](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)  
* **Classes:** Battery, Keyboard, Microwave, Mobile, Mouse, PCB, Player, Printer, Television, Washing Machine
* **Structure:** Organized in folders by class label.
* **Size:** Balanced with 2400 training images, 300 validation images, 300 test images.

---

## 🔄 Project Workflow

### 1. Dataset Preparation

* Unzipped and structured into train, validation, and test folders.
* Used `image_dataset_from_directory()` to load and preprocess the images.
* Checked for class balance and distribution via bar plots.

### 2. Data Augmentation

Applied the following to improve generalization:

* `RandomFlip(horizontal)`
* `RandomRotation(0.1)`
* `RandomZoom(0.1)`

### 3. Model Architecture

* **Base Model:** EfficientNetV2B0 with imagenet weights.
* **Customization:**
    * Frozen first 100 layers of base model.
    * Added: `GlobalAveragePooling2D`, `Dropout(0.2)`, and `Dense(10, softmax)`.
* **Compiled using:**
    * `optimizer = Adam(learning_rate=0.0001)`
    * `loss = SparseCategoricalCrossentropy()`
    * `metrics = ['accuracy']`

### 4. Training

* Trained on Google Colab using `fit()` with EarlyStopping.
* **Epochs:** 8
* Achieved 96% accuracy on the test dataset.

---

## 📊 Results and Evaluation

* **Test Accuracy:** 96.00%
* **Confusion Matrix:** Confirmed strong classification across all 10 categories.
* **Classification Report:** High precision, recall, and F1-scores for all classes.
* **Graphs:**
    * Training vs Validation Accuracy
    * Training vs Validation Loss

---

## 🌐 Deployment

* Built a simple Gradio Web Interface:
    * User can upload any image of e-waste.
    * The model displays predicted class and confidence.
* Clean and modern UI for public demonstration.

---

## 🚀 Key Improvements Over Time

* Switched from basic CNN to EfficientNetV2B0 for higher performance.
* Optimized augmentation and learning rate for better generalization.
* Cleaned dataset loading and validation flow.
* Visualized predictions using sample images with predicted vs actual labels.
* Integrated Gradio for real-time interaction and testing.

---

## 📈 Future Scope

* Mobile app integration for personal e-waste classification.
* Real-time edge deployment using Raspberry Pi or IoT devices.
* Extension to more categories and real-world mixed environments.
* Integration with city-level recycling systems for smart waste sorting.

---

## 👨‍💻 Author

Aniket Redekar
BE. CSE, 2nd Year
AI/ML Virtual Internship – Edunet Foundation
July 2025
