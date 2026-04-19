import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# =========================
# Load model + vectorizer
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("🧠 Sentiment Analysis Dashboard")
st.write("Analyze product reviews using Machine Learning")

# =========================
# Preprocessing
# =========================
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# =========================
# Single Review Prediction
# =========================
st.subheader("🔍 Analyze Single Review")

review = st.text_area("Enter your review:")

if st.button("Analyze Review"):
    if review.strip() != "":
        clean = preprocess_text(review)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]
        probs = model.predict_proba(vec)
        confidence = max(probs[0])

        st.success(f"Sentiment: {pred}")
        st.info(f"Confidence: {round(confidence, 2)}")
    else:
        st.warning("Please enter a review")

# =========================
# CSV Upload Section
# =========================
st.subheader("📂 Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Analyze File"):
        df["cleaned"] = df.iloc[:, 0].apply(preprocess_text)
        df["Prediction"] = model.predict(vectorizer.transform(df["cleaned"]))

        st.write("Results:")
        st.dataframe(df)

        # Visualization
        st.subheader("📊 Sentiment Distribution")
        counts = df["Prediction"].value_counts()

        fig = px.pie(values=counts.values, names=counts.index, title="Sentiment Breakdown")
        st.plotly_chart(fig)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
