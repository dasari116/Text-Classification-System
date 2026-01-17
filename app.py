import streamlit as st
from model import load_data, train_model, predict_requirement

st.set_page_config(
    page_title="Software Requirement Classifier",
    layout="centered"
)

st.title("Software Requirements Classification System")

st.markdown("""
This application classifies software requirements using  
**multiple machine learning models** trained with TF-IDF features.
""")

# Load data
@st.cache_data
def load_dataset():
    return load_data("Dataset.csv")

df = load_dataset()

# Model selection
model_name = st.selectbox(
    "Select Machine Learning Model",
    ["Naive Bayes", "SVM", "Random Forest", "Decision Tree", "MLP"]
)

# Train model once
@st.cache_resource
def load_model(model_name):
    return train_model(df, model_name)

model, vectorizer = load_model(model_name)

# User input
user_text = st.text_area(
    "Enter Software Requirement Text",
    height=150
)

if st.button("Classify Requirement"):
    if user_text.strip() == "":
        st.warning("Please enter a requirement description.")
    else:
        prediction = predict_requirement(
            user_text, model, vectorizer
        )
        st.success(f"Predicted Category: **{prediction}**")
