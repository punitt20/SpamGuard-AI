"""Flask application factory for the local SMS spam analyzer."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request

from config import Config


def _configure_logging(app: Flask) -> None:
    """Log to the console and a rotating file without duplicating handlers."""
    if getattr(app, "_spamguard_logging_configured", False):
        return
    app.logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if not app.logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        app.logger.addHandler(stream)

    try:
        app.config["LOG_DIR"].mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            app.config["LOG_DIR"] / "spamguard.log",
            maxBytes=500_000,
            backupCount=2,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
    except OSError:
        app.logger.warning("Could not initialise file logging; using console only.")

    app._spamguard_logging_configured = True  # type: ignore[attr-defined]


def create_app(
    config_object: type[Config] | None = None,
    test_config: dict | None = None,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        instance_path=str((config_object or Config).INSTANCE_DIR),
        instance_relative_config=False,
        static_folder="static",
        template_folder="templates",
    )
    app.config.from_object(config_object or Config)
    if test_config:
        app.config.update(test_config)

    # Keep JSON key order stable for readable responses.
    app.json.sort_keys = False

    for path_key in ("MODEL_DIR", "INSTANCE_DIR", "LOG_DIR"):
        app.config[path_key].mkdir(parents=True, exist_ok=True)

    _configure_logging(app)

    # Lazy model loading. The model file is only read on the first prediction.
    from app.services.model_registry import ModelRegistry
    from app.services.prediction_service import PredictionService

    registry_specs = {
        "spam": {
            "model_path": app.config["SPAM_MODEL_PATH"],
            "metadata_path": app.config["SPAM_METADATA_PATH"],
            "label_positive": "SPAM",
            "label_negative": "HAM",
        },
        "phishing": {
            "model_path": app.config["PHISHING_MODEL_PATH"],
            "metadata_path": app.config["PHISHING_METADATA_PATH"],
            "label_positive": "PHISHING",
            "label_negative": "LEGITIMATE",
        },
    }

    registry = ModelRegistry(registry_specs)
    app.extensions["model_registry"] = registry
    app.extensions["prediction_service"] = PredictionService(
        registry=registry,
        max_message_length=app.config["MAX_MESSAGE_LENGTH"],
    )

    # HTML page routes and API blueprints.
    from app.routes.main import main_bp
    from app.routes.prediction import prediction_bp
    from app.routes.model_info import model_info_bp
    from app.routes.analytics import analytics_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp, url_prefix="/api")
    app.register_blueprint(model_info_bp, url_prefix="/api/model")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")

    # Error handling -----------------------------------------------------
    @app.errorhandler(413)
    def payload_too_large(_error):
        return _error_response("The request body is too large.", 413)

    @app.errorhandler(404)
    def not_found(_error):
        return _error_response("The requested resource was not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return _error_response("This HTTP method is not allowed for the endpoint.", 405)

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.exception("Unhandled application error: %s", error)
        return _error_response("An unexpected server error occurred.", 500)

    app.logger.info(
        "SpamGuard AI started (env=%s, sms_model=%s, phishing_model=%s)",
        os.getenv("FLASK_ENV", "local"),
        app.config["SPAM_MODEL_PATH"],
        app.config["PHISHING_MODEL_PATH"],
    )
    return app


def _error_response(message: str, status: int):
    """Return JSON for API paths and a tiny HTML page for browser paths."""
    if request.path.startswith("/api/"):
        return jsonify({"error": message, "status": status}), status
    return (
        "<!doctype html><html><head><title>SpamGuard AI</title>"
        "<link rel='stylesheet' href='/static/css/app.css'></head>"
        "<body><main class='app-main'><h1>"
        f"{status}</h1><p>{message}</p><a href='/'>Return to the analyzer</a>"
        "</main></body></html>",
        status,
    )
