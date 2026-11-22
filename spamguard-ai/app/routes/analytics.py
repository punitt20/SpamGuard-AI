"""Read-only evaluation summaries from saved model metadata."""
from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request, abort

analytics_bp = Blueprint("analytics", __name__)
SUPPORTED_TASKS = ("all", "spam", "phishing")


@analytics_bp.get("")
def analytics():
    """Return candidate evaluation results; no message history is stored."""
    task = (request.args.get("task") or "all").lower()
    if task not in SUPPORTED_TASKS:
        abort(400, description="Invalid task. Use 'all', 'spam', or 'phishing'.")

    registry = current_app.extensions["model_registry"]
    model_comparison = {}
    active_models = []

    for task_name in ("spam", "phishing"):
        if task not in ("all", task_name):
            continue
        info = registry.info(task_name)
        if not info.get("available"):
            continue

        # ModelRegistry.info returns the saved metadata at the top level.
        # Keep this endpoint aligned with that shape so candidate metrics are
        # available to the analytics page.
        metadata = info
        model_comparison[task_name] = metadata.get("comparison", [])
        dataset = metadata.get("dataset", {})
        active_models.append(
            {
                "task": task_name,
                "model_name": info.get("model_name"),
                "model_version": info.get("model_version"),
                "trained_at": info.get("trained_at"),
                "dataset_source": dataset.get("source", "")
                if isinstance(dataset, dict)
                else "",
                "dataset": dataset,
                "metrics": info.get("metrics", {}),
                "classification_threshold": info.get("classification_threshold"),
                "selection_criterion": info.get("selection_criterion", ""),
                "confusion_matrix": info.get("confusion_matrix", {}),
                "features": info.get("features", {}),
                "preprocessing": info.get("preprocessing", {}),
            }
        )

    return jsonify(
        {
            "filter": task,
            "model_comparison": model_comparison,
            "active_models": active_models,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "Evaluation metrics are read from saved training metadata. Message history is not stored.",
        }
    )
