import streamlit as st
import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("📧 Spam Detection App")

message = st.text_area("Enter a message")

if st.button("Predict"):

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("🚨 SPAM")
    else:
        st.success("✅ HAM")