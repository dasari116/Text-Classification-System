# Text-Classification-System
A Streamlit-based machine learning application for classifying software requirements using TF-IDF features and multiple classifiers including Naive Bayes, SVM, Random Forest, Decision Tree, and MLP.

Software Requirements Classification using Machine Learning

📌 Project Overview
This project implements a machine learning–based text classification system that classifies software requirement descriptions into predefined Requirement Types.
The system is built using classical machine learning algorithms and deployed through a Streamlit web interface for interactive prediction.


📂 Dataset

File: Dataset.csv
Text Feature: Requirement
Target Label: Requirement Type
Additional Columns: Scenario, Author (not used in training)

⚙️ Technologies Used

Python
Pandas
Scikit-learn
Streamlit
TF-IDF Vectorization

🧠 Methods and Approach

1. Data Loading
Dataset is loaded using Pandas
Column names are cleaned using str.strip() to avoid schema issues
2. Text Feature Extraction
TF-IDF Vectorizer is used to convert requirement text into numerical features
English stop words are removed
Both unigrams and bigrams are considered
Maximum document frequency is limited (max_df = 0.95) to reduce noise
3. Machine Learning Models Used
The following models are implemented and selectable from the UI:
Multinomial Naive Bayes
Support Vector Machine (Linear Kernel)
Random Forest Classifier
Decision Tree Classifier
Multi-Layer Perceptron (Neural Network)

4. Model Training

Data is split into 80% training and 20% testing
Stratified sampling is applied to maintain class distribution
Selected model is trained on TF-IDF features
Model training is cached using Streamlit to improve performance

5. Prediction

User enters a software requirement text
Text is transformed using the trained TF-IDF vectorizer
Selected model predicts the Requirement Type
Prediction is displayed instantly in the Streamlit interface

📊 Output

Predicted requirement category based on the selected machine learning model
Real-time response through a web-based UI

📌 Key Learnings

Importance of text preprocessing in NLP tasks
Effectiveness of TF-IDF for requirement classification
Performance differences between multiple ML algorithms
Separation of ML logic and UI improves scalability and deployment

🚀 Future Improvements

Display evaluation metrics (Accuracy, Precision, Recall, F1-score)
Add model comparison visualization
Enable dataset upload functionality
Save and load trained models

Deploy the application to a cloud platform
