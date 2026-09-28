# Spam Detection MLOps Project

## Overview

This project is a machine learning application that classifies text messages as **SPAM** or **HAM**.

The project demonstrates a complete workflow from data preprocessing and model training to experiment tracking, API deployment, and an interactive web application.

## Features

* Text preprocessing using **TF-IDF**
* **Logistic Regression** classification
* Model training and evaluation using scikit-learn
* **97.76% test accuracy**
* **MLflow** experiment tracking
* Saved trained model and TF-IDF vectorizer
* **FastAPI** REST API for predictions
* **Streamlit** web application
* Prediction confidence display
* Prediction history in the Streamlit application
* GitHub version control

## Project Structure

```text
spam detection machine learning ops project/
│
├── api/
│   └── app.py
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── src/
│   └── train.py
│
├── mlruns/
│   └── MLflow experiment data
│
├── app.py
├── requirements.txt
├── README.md
└── venv/
```

## Machine Learning Pipeline

```text
Spam Dataset
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Save Model + Vectorizer
     ↓
MLflow Tracking
     ↓
FastAPI / Streamlit
     ↓
SPAM or HAM Prediction
```

## Model

The project uses:

**TF-IDF Vectorizer**

TF-IDF converts text messages into numerical features that the machine learning model can process.

**Logistic Regression**

A Logistic Regression classifier is trained on the TF-IDF features to classify messages as SPAM or HAM.

### Test Accuracy

**97.76%**

The model was trained and tested locally using a train/test split with `random_state=42`.

## MLflow

MLflow is used to track the machine learning experiment.

The training process records:

* Model type
* Test accuracy
* Trained model artifact

The trained model is also saved locally for inference.

## FastAPI

The trained model is exposed through a FastAPI REST API.

Example endpoint:

```text
GET /predict
```

Example:

```text
Input:
Free iPhone offer! Click now to claim your prize!

Output:
SPAM
```

A normal message was also tested successfully:

```text
Input:
Hey, are we still meeting for lunch today?

Output:
HAM
```

Both API tests returned HTTP 200 responses.

## Streamlit Application

The project also includes an interactive Streamlit web application.

The application allows users to:

1. Enter a message
2. Run the trained model
3. View the SPAM/HAM prediction
4. View prediction confidence
5. View prediction history

Example:

```text
Message:
can you come tomorrow for meeting at 11am

Prediction:
HAM

Confidence:
97.50%

Confidence Level:
Very High Confidence
```

## Running the Project

### 1. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 2. Train the model

```bash
python src/train.py
```

This trains the model, records the experiment with MLflow, and saves:

```text
models/spam_model.pkl
models/vectorizer.pkl
```

### 3. Run FastAPI

```bash
uvicorn api.app:app --reload
```

Open the FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the web application can be opened in the browser.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* FastAPI
* Uvicorn
* Streamlit
* MLflow
* GitHub

## Project Status

The complete local workflow has been tested successfully:

```text
Training             ✅
Model saving         ✅
MLflow tracking      ✅
FastAPI              ✅
API SPAM prediction  ✅
API HAM prediction   ✅
Streamlit            ✅
Web prediction       ✅
```

## Future Improvements

Potential future improvements include:

* Docker containerization
* Automated testing
* CI/CD with GitHub Actions
* Prometheus and Grafana monitoring
* Cloud deployment
* Automated model retraining
* Model and data drift monitoring
