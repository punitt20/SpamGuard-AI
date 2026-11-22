"""Shared pytest fixtures for the local prediction API."""
from __future__ import annotations

from pathlib import Path

import pytest

from app import create_app
from config import Config

SPAM_READY = Config.SPAM_MODEL_PATH.exists() and Config.SPAM_METADATA_PATH.exists()
PHISHING_READY = (
    Config.PHISHING_MODEL_PATH.exists() and Config.PHISHING_METADATA_PATH.exists()
)

requires_spam_model = pytest.mark.skipif(
    not SPAM_READY,
    reason="SMS model artefacts are absent; run python training/train.py --task spam.",
)


@pytest.fixture()
def app_factory(tmp_path: Path):
    """Return a function that builds isolated test applications."""

    def factory(**overrides):
        test_config = {
            "TESTING": True,
            "INSTANCE_DIR": tmp_path / "instance",
            "LOG_DIR": tmp_path / "instance" / "logs",
        }
        test_config.update(overrides)
        return create_app(test_config=test_config)

    return factory


@pytest.fixture()
def app(app_factory):
    application = app_factory()
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def spam_sample() -> str:
    return (
        "Congratulations! You have won a FREE cash prize of ₹50,000. "
        "Claim it immediately at http://bit.ly/claim-prize and send your OTP."
    )


@pytest.fixture()
def ham_sample() -> str:
    return "Hi, are we still meeting near the station at 6 pm today?"


@pytest.fixture()
def post_prediction(client):
    """Small helper that posts one message and returns the JSON response."""

    def post(message: str, task: str = "spam"):
        response = client.post("/api/predict", json={"message": message, "task": task})
        assert response.status_code == 200, response.get_data(as_text=True)
        return response.get_json()

    return post
