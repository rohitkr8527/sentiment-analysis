import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings

from transformers import AutoTokenizer, AutoModel
import torch

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

CONFIG = {
    "data_path": "notebooks/balanced_sentiment_dataset.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/rohitkr8527/sentiment-analysis.mlflow",
    "dagshub_repo_owner": "rohitkr8527",
    "dagshub_repo_name": "sentiment-analysis",
    "experiment_name": "Embedding Vectorizers with Logistic Regression"
}

#  SETUP MLflow & DAGSHUB 
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

#  TEXT PREPROCESSING 
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
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_extra_whitespace(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(df):
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

# --------- Embedding Loading & Vectorizers ---------

print("Loading Transformer model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
transformer_model = AutoModel.from_pretrained('distilbert-base-uncased')
transformer_model.eval()

def embedding_vectorizer(texts, embedding_dict, embed_dim):
    vectors = []
    for text in texts:
        words = text.split()
        word_vectors = [embedding_dict[word] for word in words if word in embedding_dict]
        if word_vectors:
            avg_vector = np.mean(word_vectors, axis=0)
        else:
            avg_vector = np.zeros(embed_dim)
        vectors.append(avg_vector)
    return np.vstack(vectors)



def transformer_vectorizer(texts):
    vectors = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            outputs = transformer_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            vectors.append(cls_emb)
    return np.vstack(vectors)

from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZERS = {
    'TF-IDF': lambda texts: TfidfVectorizer(max_features=5000).fit_transform(texts),
    'Transformer': transformer_vectorizer
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

def log_model_params(algo_name, model):
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    mlflow.log_params(params_to_log)

def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer_func in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        print(f"\nRunning {algo_name} with {vec_name} vectorizer...")
                        X = vectorizer_func(df['text'].tolist())
                        y = df['sentiment']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        model = algorithm
                        model.fit(X_train, y_train)

                        log_model_params(algo_name, model)

                        y_pred = model.predict(X_test)
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        mlflow.log_metrics(metrics)

                        input_example = X_test[:5]
                        if scipy.sparse.issparse(X_test):
                            input_example = input_example.toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}: {e}")
                        mlflow.log_param("error", str(e))

if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
