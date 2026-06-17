from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}


@app.get("/predict")
def predict(text: str):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return {"prediction": result}