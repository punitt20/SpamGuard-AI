"""SMS spam prediction API endpoint."""
from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from app.services.model_registry import ModelUnavailableError
from app.services.prediction_service import PredictionInputError, PredictionService

prediction_bp = Blueprint("prediction", __name__)
LOGGER = logging.getLogger(__name__)


def _prediction_service() -> PredictionService:
    return current_app.extensions["prediction_service"]


def _error(message: str, status: int, code: str):
    return jsonify({"error": message, "code": code, "status": status}), status


@prediction_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return _error("Request body must be valid JSON with a 'message' field.", 400, "invalid_json")
    if not isinstance(payload, dict):
        return _error("JSON body must be an object.", 400, "invalid_json")

    task = payload.get("task", "spam")
    if not isinstance(task, str):
        return _error("The 'task' field must be a string.", 400, "invalid_task")

    try:
        analysis = _prediction_service().analyze(payload.get("message"), task=task)
    except PredictionInputError as exc:
        return _error(str(exc), 400, "invalid_message")
    except ModelUnavailableError as exc:
        return _error(str(exc), 503, "model_unavailable")

    LOGGER.info(
        "Single prediction task=%s class=%s length=%s",
        task,
        analysis["classification"],
        len(analysis["message"]),
    )
    return jsonify(analysis), 200
