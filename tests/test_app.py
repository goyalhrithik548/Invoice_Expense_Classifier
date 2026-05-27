"""
Unit tests for the Invoice Expense Classifier.
Run with: pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from app.preprocessor import preprocess


# --- Preprocessor Tests ---

def test_preprocess_lowercases():
    assert preprocess("AWS Monthly BILL") == "aws monthly bill"


def test_preprocess_removes_special_chars():
    result = preprocess("invoice #1234 @ $500!")
    assert "#" not in result
    assert "$" not in result
    assert "!" not in result


def test_preprocess_collapses_whitespace():
    result = preprocess("  too   many   spaces  ")
    assert result == "too many spaces"


def test_preprocess_empty_string():
    assert preprocess("") == ""


# --- API Tests (require trained model) ---

MODEL_EXISTS = os.path.exists(os.path.join("model", "model.pkl"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_predict_endpoint_returns_category():
    from main import app
    client = TestClient(app)
    response = client.post("/predict", json={"text": "AWS monthly cloud hosting bill"})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert data["category"] == "Cloud/Software"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_predict_endpoint_returns_confidence():
    from main import app
    client = TestClient(app)
    response = client.post("/predict", json={"text": "Blue Dart courier charges"})
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_predict_empty_text_returns_400():
    from main import app
    client = TestClient(app)
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_predict_logistics():
    from main import app
    client = TestClient(app)
    response = client.post("/predict", json={"text": "FedEx freight delivery charges"})
    assert response.status_code == 200
    assert response.json()["category"] == "Logistics"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_predict_travel():
    from main import app
    client = TestClient(app)
    response = client.post("/predict", json={"text": "Flight booking for client meeting in Delhi"})
    assert response.status_code == 200
    assert response.json()["category"] == "Travel"


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet. Run train.py first.")
def test_health_check():
    from main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
