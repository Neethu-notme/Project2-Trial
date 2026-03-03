import streamlit as st
import joblib
import numpy as np
import re
from docx import Document
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------
# LOAD MODEL & VECTORIZER
# ----------------------------
model = joblib.load("svm_rbf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Resume Classifier", layout="centered")

st.title("📄 Resume Category Predictor")

# ----------------------------
# CLEANING FUNCTION (EDA + NLP)
# ----------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # remove numbers & punctuation
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # remove stopwords
    return " ".join(words)

# ----------------------------
# EXTRACT TEXT FROM DOCX
# ----------------------------
def extract_text(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Resume (.docx file)",
    type=["docx"]
)

if uploaded_file is not None:

    raw_text = extract_text(uploaded_file)

    if raw_text.strip() == "":
        st.warning("Uploaded file is empty.")
    else:
        # Clean text
        cleaned_text = clean_text(raw_text)

        # Transform
        text_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Confidence
        try:
            probabilities = model.predict_proba(text_vector)
            confidence = np.max(probabilities) * 100
        except:
            confidence = "Model not trained with probability=True"

        # Display
        st.success(f"Predicted Category: {prediction}")
        st.info(f"Confidence: {confidence if isinstance(confidence, str) else f'{confidence:.2f}%'}")

        # Optional: Show cleaned text
        with st.expander("See Cleaned Text"):
            st.write(cleaned_text)
