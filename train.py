"""
Training script — run this once to produce model/model.pkl
Usage: python train.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from app.preprocessor import preprocess

DATA_PATH = os.path.join("data", "invoices.csv")
MODEL_PATH = os.path.join("model", "model.pkl")


def load_data(path: str):
    df = pd.read_csv(path)
    df["text"] = df["text"].apply(preprocess)
    return df["text"].tolist(), df["category"].tolist()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
    ])


def train():
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)} samples.")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
