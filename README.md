\# Spam Detection MLOps Project



\## Overview



This project detects whether a message is SPAM or HAM using Machine Learning.



The project includes:



\* Data preprocessing using TF-IDF

\* Logistic Regression model

\* FastAPI REST API

\* Streamlit Web Application

\* MLflow Experiment Tracking

\* GitHub Version Control



\## Project Structure



```text

spam-detection-mlops/

│

├── api/

│   └── app.py

│

├── data/

│   └── spam.csv

│

├── models/

│   ├── spam\_model.pkl

│   └── vectorizer.pkl

│

├── src/

│   └── train.py

│

├── app.py

├── requirements.txt

└── README.md

```



\## Model



\* TF-IDF Vectorizer

\* Logistic Regression Classifier



Accuracy achieved: 96%



\## Running the Project



\### Train Model



```bash

python src/train.py

```



\### Run FastAPI



```bash

uvicorn api.app:app --reload

```



\### Run Streamlit



```bash

streamlit run app.py

```



\## Future Improvements



\* Docker

\* GitHub Actions

\* Automated Testing

\* Cloud Deployment

\* Monitoring with Prometheus and Grafana



