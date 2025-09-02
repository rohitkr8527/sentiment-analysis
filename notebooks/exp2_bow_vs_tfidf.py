import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Download required nltk data
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

CONFIG = {
    "data_path": "notebooks/balanced_sentiment_dataset.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/rohitkr8527/sentiment-analysis.mlflow",
    "dagshub_repo_owner": "rohitkr8527",
    "dagshub_repo_name": "sentiment-analysis",
    "experiment_name": "Bow vs TfIdf"
}

#  SETUP MLflow & DAGSHUB 
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

#  TEXT PREPROCESSING 
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_extra_whitespace(text):
    """Remove extra whitespace."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['text'] = df['text'].apply(lower_case)
        df['text'] = df['text'].apply(removing_urls)
        df['text'] = df['text'].apply(removing_numbers)
        df['text'] = df['text'].apply(removing_punctuations)
        df['text'] = df['text'].apply(remove_stop_words)
        df['text'] = df['text'].apply(lemmatization)
        df['text'] = df['text'].apply(remove_extra_whitespace)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise

#  LOAD & PREPROCESS DATA
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

#  FEATURE ENGINEERING
VECTORIZERS = {
    'BoW': CountVectorizer(max_features=5000),
    'TF-IDF': TfidfVectorizer(max_features=5000)
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

#  TRAIN & EVALUATE MODELS
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # Feature extraction
                        X = vectorizer.fit_transform(df['text'])
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        # Log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # Train model
                        model = algorithm
                        model.fit(X_train, y_train)

                        # Log model parameters
                        log_model_params(algo_name, model)

                        # Evaluate model
                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        mlflow.log_metrics(metrics)

                        # Log model
                        input_example = X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # Print results for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        # Using get_params for safer access
        params = model.get_params()
        params_to_log["n_estimators"] = params.get("n_estimators", None)
        params_to_log["learning_rate"] = params.get("learning_rate", None)
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)

# EXECUTION 
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
