# CovDetect
A deep learning model that analyzes chest X-ray images to distinguish between COVID-19 infected lungs and normal lungs. Built using Convolutional Neural Networks (CNN) for rapid, accurate respiratory disease screening.

# 🫁 CovDetect – COVID vs Normal Lung X-ray Classifier

LungScanAI is a deep learning project that uses Convolutional Neural Networks (CNNs) to classify chest X-ray (CXR) images as either **COVID-19 infected lungs** or **normal lungs**. This tool aims to support rapid respiratory screening, especially in resource-limited settings.

## 🚀 Project Overview

- Trained a CNN model on publicly available COVID-19 CXR datasets
- Performed image preprocessing, data augmentation, and binary classification
- Evaluated model performance using accuracy

## 🧠 Model Architecture

- Convolutional Layers with ReLU activations
- Max Pooling for feature reduction
- Dense Layer(s) with Dropout for regularization
- Sigmoid Output for binary classification

  ## ✅ Pretrained Model (Transfer Learning)
- **MobileNetV2** with custom classification head
- Fine-tuned for COVID-vs-Normal classification


## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn

## 📊 Dataset

The dataset includes labeled COVID-19 and Normal lung chest X-ray images collected from open public sources.


## 📈 Evaluation Metrics

- Accuracy
- Precision / Recall

