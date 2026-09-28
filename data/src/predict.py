import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

message = input("Enter message: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)

probability = model.predict_proba(message_vector)[0]
confidence = max(probability) * 100

print(f"Confidence: {confidence:.2f}%")

if prediction[0] == 1:
    print("SPAM")
else:
    print("HAM")
