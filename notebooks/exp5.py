import os
import re
import string
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# nltk.download('stopwords')
# nltk.download('wordnet')

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "https://dagshub.com/rohitkr8527/sentiment-analysis.mlflow"
dagshub.init(repo_owner="rohitkr8527", repo_name="sentiment-analysis", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Tf-Idf Hyperparameter Tuning")

def preprocess_text(text):
    """
    Cleans and preprocesses input text by applying:
    - Lowercasing
    - Number and punctuation removal
    - Stopwords removal
    - Lemmatization
    - URL removal
    - Extra whitespace cleanup

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned and preprocessed text
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopwords removal
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces

    # logging.debug(f"Processed text preview: {text[:100]}...")
    return text

def load_and_prepare_data(filepath):
    """
    Loads dataset from CSV, applies text preprocessing, and splits into train/test sets.

    Args:
        filepath (str): Path to the CSV dataset

    Returns:
        tuple: X_train, X_test, y_train, y_test (preprocessed)
    """
    logging.info("Loading dataset...")
    df = pd.read_csv(filepath)

    logging.info("Preprocessing text...")
    df["text"] = df["text"].astype(str).apply(preprocess_text)
    X = df["text"]
    y = df["sentiment"]

    logging.info(f"Total samples: {len(X)}")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_vectorizer_model(X_train_texts, X_test_texts, y_train, y_test):
    """
    Performs hyperparameter tuning for TfidfVectorizer using GridSearchCV,
    logs metrics and models to MLflow. LogisticRegression hyperparameters are fixed.

    Args:
        X_train_texts (Series): Training text data
        X_test_texts (Series): Testing text data
        y_train (Series): Training labels
        y_test (Series): Testing labels
    """
    logging.info("Setting up pipeline and parameter grid...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=1,
            max_iter=1000
        ))
    ])

    param_grid = {
        "tfidf__max_df": [0.9, 1.0],
        "tfidf__min_df": [1, 5],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [5000, 10000, None]
    }

    with mlflow.start_run():
        logging.info("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_texts, y_train)

        for i, (params, mean_score, std_score) in enumerate(zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"]
        )):
            logging.debug(f"Run {i+1}: Params: {params}, Mean F1: {mean_score:.4f}, Std: {std_score:.4f}")

            with mlflow.start_run(run_name=f"TFIDF Params: {params}", nested=True):
                pipeline.set_params(**params)
                pipeline.fit(X_train_texts, y_train)
                y_pred = pipeline.predict(X_test_texts)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                # Log TF-IDF params
                mlflow.log_params(params)

                # Log fixed LR params
                mlflow.log_params({
                    "clf__solver": "saga",
                    "clf__penalty": "elasticnet",
                    "clf__C": 1,
                    "clf__l1_ratio": 0.5
                })

                # Log metrics
                mlflow.log_metrics(metrics)

                logging.info(f"Logged metrics for TF-IDF params: {params}")
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        #  Best model & parameter logging
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        # Merge TF-IDF and fixed LR params
        vectorizer_params = {k: v for k, v in best_params.items() if k.startswith("tfidf__")}
        fixed_lr_params = {
            "clf__solver": "saga",
            "clf__penalty": "elasticnet",
            "clf__C": 1,
            "clf__l1_ratio": 0.5
        }
        all_logged_params = {**vectorizer_params, **fixed_lr_params}

        mlflow.log_params(all_logged_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")

        logging.info(f"Best Params: {all_logged_params}")
        print(f"\nBest TF-IDF Params: {vectorizer_params} | Best F1 Score: {best_f1:.4f}")

if __name__ == "__main__":
    """
    Loads data, trains the pipeline with TF-IDF tuning and fixed Logistic Regression,
    and logs results to MLflow and DAGsHub.
    """
    X_train_texts, X_test_texts, y_train, y_test = load_and_prepare_data("notebooks/balanced_sentiment_dataset.csv")
    train_and_log_vectorizer_model(X_train_texts, X_test_texts, y_train, y_test)
