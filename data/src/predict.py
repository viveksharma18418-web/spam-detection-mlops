import joblib

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

message = input("Enter message: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)

if prediction[0] == 1:
    print("SPAM")
else:
    print("HAM")