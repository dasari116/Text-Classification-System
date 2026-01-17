import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "MLP": MLPClassifier(max_iter=300, random_state=42)
    }

def train_model(df, model_name):
    # Text and label columns (matched to your CSV)
    X = df["Requirement"]
    y = df["Requirement Type"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95
    )

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = get_models()[model_name]
    model.fit(X_train, y_train)

    return model, vectorizer

def predict_requirement(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
