"""Application configuration.

Paths are resolved relative to the project root so the app can be started from
any working directory. The web app uses the SMS model paths and message limit;
the remaining paths are used by the optional offline training/evaluation scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Default configuration for the Flask app and offline training scripts."""

    BASE_DIR = BASE_DIR
    APP_VERSION = "1.1.0"

    # Directories
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"
    REPORT_DIR = BASE_DIR / "reports"
    FIGURE_DIR = REPORT_DIR / "figures"
    INSTANCE_DIR = BASE_DIR / "instance"
    LOG_DIR = INSTANCE_DIR / "logs"

    # Saved SMS model
    SPAM_MODEL_PATH = MODEL_DIR / "spam_classifier.pkl"
    SPAM_METADATA_PATH = MODEL_DIR / "model_metadata.json"

    # Optional phishing training paths kept for the offline trainer/evaluator.
    # The web app does not expose or serve a phishing model.
    PHISHING_MODEL_PATH = MODEL_DIR / "phishing_classifier.pkl"
    PHISHING_METADATA_PATH = MODEL_DIR / "phishing_metadata.json"

    # Input limits
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB request body
    MAX_MESSAGE_LENGTH = 20_000
    MODEL_MAX_TEXT_CHARS = 20_000

    # Application behaviour
    TESTING = False
    DEBUG = _env_bool("SPAMGUARD_DEBUG", False)


class TestConfig(Config):
    TESTING = True
