import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# App title
st.title("Spam Detection App")

st.write(
    "Enter a message below and the model will predict whether it is SPAM or HAM."
)

# User input
message = st.text_area("Enter a message")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        vector = vectorizer.transform([message])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.error("SPAM")
        else:
            st.success("HAM")

        st.info(f"Confidence: {confidence:.2f}%")

        result = "SPAM" if prediction == 1 else "HAM"

        st.session_state.history.append(
            {
                "message": message,
                "result": result,
                "confidence": f"{confidence:.2f}%"
            }
        )

        if confidence > 90:
            st.success("Very High Confidence")
        elif confidence > 75:
            st.success("High Confidence")
        elif confidence > 60:
            st.warning("Moderate Confidence")
        else:
            st.warning("Low Confidence")

# Prediction history
st.subheader("Prediction History")

for item in reversed(st.session_state.history):
    st.write(
        f"Message: {item['message']} | "
        f"Result: {item['result']} | "
        f"Confidence: {item['confidence']}"
    )