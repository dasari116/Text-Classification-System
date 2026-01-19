# Text-Classification-System

A Streamlit-based machine learning application for classifying software requirements using TF-IDF features and multiple classical ML classifiers.

---

## 📌 Project Overview

This project implements a **machine learning–based text classification system** that classifies software requirement descriptions into predefined **Requirement Types**.

The system uses classical NLP and ML techniques and is deployed using a **Streamlit web interface** to allow interactive, real-time predictions.

---

## 📂 Dataset

* **File Name:** `Dataset.csv`
* **Text Feature:** `Requirement`
* **Target Label:** `Requirement Type`
* **Additional Columns:** `Scenario`, `Author` (not used for training)

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* TF-IDF Vectorization

---

## 🧠 Methods and Approach

### 1. Data Loading

* Dataset loaded using Pandas
* Column names cleaned using `str.strip()` to avoid schema mismatches

### 2. Text Feature Extraction

* TF-IDF Vectorizer for converting text into numerical features
* English stop words removed
* Unigrams and bigrams used
* `max_df = 0.95` applied to reduce noisy terms

### 3. Machine Learning Models

The following models are implemented and selectable via the Streamlit UI:

* Multinomial Naive Bayes
* Support Vector Machine (Linear Kernel)
* Random Forest Classifier
* Decision Tree Classifier
* Multi-Layer Perceptron (Neural Network)

### 4. Model Training

* Data split: **80% training / 20% testing**
* Stratified sampling to preserve class distribution
* Model trained on TF-IDF features
* Training cached using Streamlit for performance optimization

### 5. Prediction Workflow

* User inputs a software requirement text
* Text transformed using trained TF-IDF vectorizer
* Selected model predicts the **Requirement Type**
* Result displayed instantly in the Streamlit interface

---

## 📊 Output

* Predicted requirement category based on the selected ML model
* Real-time prediction through a web-based UI

---

## 📌 Key Learnings

* Importance of text preprocessing in NLP tasks
* Effectiveness of TF-IDF for software requirement classification
* Performance variation across different ML algorithms
* Clear separation of ML logic and UI improves scalability

---

## 🚀 Future Improvements

* Display evaluation metrics (Accuracy, Precision, Recall, F1-score)
* Add model comparison visualizations
* Enable dataset upload through the UI
* Save and load trained models
* Deploy the application to a cloud platform
