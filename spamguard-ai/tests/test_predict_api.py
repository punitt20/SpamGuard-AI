"""Core happy-path tests for POST /api/predict on the simplified single-endpoint app.

Covers:
- valid input (spam message returns a 200 classification)
- invalid JSON (400 invalid_json)
- empty message (400 invalid_message)
- model unavailable (503 model_unavailable)
"""
from __future__ import annotations

from tests.conftest import requires_spam_model

pytestmark = requires_spam_model


def test_predict_valid_spam_message(client, spam_sample):
    response = client.post(
        "/api/predict", json={"message": spam_sample, "task": "spam"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["classification"] == "SPAM"
    assert data["task"] == "spam"
    assert 0.0 <= data["model_probability"] <= 1.0
    assert data["classification"] == (
        "SPAM"
        if data["model_probability"] >= data["decision_threshold"]
        else "HAM"
    )
    assert data["top_signals"]["available"] is True


def test_predict_invalid_json(client):
    response = client.post(
        "/api/predict", data="{not json", content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_json"


def test_predict_empty_message(client):
    response = client.post("/api/predict", json={"message": ""})
    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_message"


def test_predict_model_unavailable(app_factory, tmp_path):
    app = app_factory(
        SPAM_MODEL_PATH=tmp_path / "missing-spam.pkl",
        SPAM_METADATA_PATH=tmp_path / "missing-spam.json",
    )
    client = app.test_client()
    response = client.post("/api/predict", json={"message": "Hello there"})
    assert response.status_code == 503
    assert response.get_json()["code"] == "model_unavailable"
