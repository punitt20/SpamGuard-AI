"""Model information API endpoints.

Exposes model metadata, available tasks, capabilities, and evaluation metrics
for both SMS spam and email phishing models.
"""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify

model_info_bp = Blueprint("model_info", __name__)


@model_info_bp.get("")
def model_info():
    """Return model registry information for all tasks."""
    registry = current_app.extensions["model_registry"]

    tasks = {}
    for task in ("spam", "phishing"):
        try:
            info = registry.info(task)
            tasks[task] = {
                "task": task,
                "available": info.get("available", False),
                "model_name": info.get("model_name"),
                "model_version": info.get("model_version"),
                "trained_at": info.get("trained_at"),
                "dataset_source": info.get("dataset", {}).get("name")
                if isinstance(info.get("dataset"), dict)
                else None,
                # ModelRegistry.info exposes the saved artifact metadata at
                # the top level, alongside its availability flag.
                "metadata": info,
            }
        except Exception as exc:  # pragma: no cover - defensive
            current_app.logger.exception("Model info error for %s", task)
            tasks[task] = {
                "task": task,
                "available": False,
                "error": str(exc),
            }

    return jsonify(
        {
            "application": "SpamGuard AI",
            "version": current_app.config.get("APP_VERSION", "1.0.0"),
            "tasks": tasks,
            "phishing_supported": tasks.get("phishing", {}).get("available", False),
            "spam_supported": tasks.get("spam", {}).get("available", False),
        }
    )


@model_info_bp.get("/<task>")
def model_info_task(task: str):
    """Return detailed model information for a specific task."""
    registry = current_app.extensions["model_registry"]

    if task not in ("spam", "phishing"):
        return jsonify({"error": f"Unknown task '{task}'", "status": 404}), 404

    try:
        info = registry.info(task)
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.exception("Model info error for %s", task)
        return jsonify({"error": str(exc), "status": 500}), 500

    if not info.get("available"):
        return jsonify(
            {
                "task": task,
                "available": False,
                "message": f"{task} model is not trained yet.",
            }
        ), 404

    metadata = info
    metrics = metadata.get("metrics", {})
    comparison = metadata.get("comparison", [])

    return jsonify(
        {
            "task": task,
            "available": True,
            "model_name": info.get("model_name"),
            "model_version": info.get("model_version"),
            "trained_at": info.get("trained_at"),
            "dataset": metadata.get("dataset", {}),
            "metrics": metrics,
            "comparison": comparison,
            "preprocessing": metadata.get("preprocessing", {}),
            "limitations": metadata.get("limitations", []),
        }
    )
