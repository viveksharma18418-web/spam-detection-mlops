import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(
    "data/spam.csv",
    sep="\t",
    encoding="latin-1"
)

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully")