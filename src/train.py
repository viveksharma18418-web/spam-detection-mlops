print("THIS IS THE NEW TRAIN FILE")
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
print("THIS IS THE NEW TRAIN FILE")

import pandas as pd

df = pd.read_csv(
    "data/spam.csv",
    sep="\t",
    encoding="latin-1"
)

print(df.head())
print(df.shape)
print("TRAIN.PY IS RUNNING")
print("Loading data...")
df = df[["v1", "v2"]]

df.columns = ["label", "message"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# MLflow tracking
with mlflow.start_run():

    vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)