from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import os
import pandas as pd
import dagshub
import numpy as np
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords")
nltk.download("wordnet")

# Text preprocessing functions 
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("sentiment_analysis")
if not dagshub_token:
    raise EnvironmentError("sentiment_analysis environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rohitkr8527"
repo_name = "sentiment-analysis"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Load MLflow model and vectorizer
model_name = "my_model"

def get_latest_model_version(model_name: str):
    client = MlflowClient()

    # Try to fetch the latest version in Production stage
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        # Fall back to the latest version in "None" stage (unregistered stage)
        versions = client.get_latest_versions(model_name, stages=["None"])

    if not versions:
        raise ValueError(f"No versions found for model '{model_name}' in any stage.")

    latest_version = versions[0]
    print(f"Using model version: {latest_version.version} in stage: {latest_version.current_stage}")
    return latest_version.version

model_version = get_latest_model_version(model_name)
model_uri = f'models:/{model_name}/{model_version}'
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# Initialize FastAPI app
app = FastAPI()

# Setup Jinja2 templates

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Prometheus metrics setup
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = templates.TemplateResponse("index.html", {"request": request, "result": None})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    # Clean text
    cleaned_text = normalize_text(text)

    # Vectorize
    features = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Metrics
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return templates.TemplateResponse("index.html", {"request": request, "result": prediction})

@app.get("/metrics")
async def metrics():
    return HTMLResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
