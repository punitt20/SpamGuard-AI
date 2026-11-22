"""Thread-safe, lazy loading of the trained joblib pipelines."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


class ModelUnavailableError(RuntimeError):
    """Raised when a requested task has no trained/saved model."""


@dataclass
class ModelBundle:
    task: str
    pipeline: Any
    metadata: dict
    label_positive: str
    label_negative: str
    loaded_from: str


class ModelRegistry:
    """Load model artefacts on first use, and reload when files change.

    The registry never fabricates a model: if an artefact is missing or cannot
    be unpickled, callers receive an explicit :class:`ModelUnavailableError`.
    """

    def __init__(self, specs: dict[str, dict]):
        self._specs = specs
        self._bundles: dict[str, ModelBundle] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def get(self, task: str) -> ModelBundle:
        if task not in self._specs:
            raise ModelUnavailableError(f"Unknown analysis task '{task}'.")
        spec = self._specs[task]
        model_path = Path(spec["model_path"])
        metadata_path = Path(spec["metadata_path"])
        if not model_path.exists() or not metadata_path.exists():
            raise ModelUnavailableError(
                f"The {task} model is not available. Run the training script first."
            )

        with self._lock:
            cached = self._bundles.get(task)
            signature = (model_path.stat().st_mtime, metadata_path.stat().st_mtime)
            if cached is not None and getattr(cached, "_signature", None) == signature:
                return cached

            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                pipeline = joblib.load(model_path)
            except Exception as exc:  # pragma: no cover - corrupt artefact path
                self._logger.exception("Failed to load %s model", task)
                raise ModelUnavailableError(
                    f"The {task} model could not be loaded. Retrain the model and try again."
                ) from exc

            if not hasattr(pipeline, "predict_proba"):
                raise ModelUnavailableError(
                    f"The saved {task} model does not support probability estimates. "
                    "Retrain with a calibrated/probability-capable classifier."
                )

            bundle = ModelBundle(
                task=task,
                pipeline=pipeline,
                metadata=metadata,
                label_positive=spec["label_positive"],
                label_negative=spec["label_negative"],
                loaded_from=str(model_path),
            )
            # Cache the file signature privately for reload-on-change.
            bundle._signature = signature  # type: ignore[attr-defined]
            self._bundles[task] = bundle
            self._logger.info(
                "Loaded %s model '%s' v%s from %s",
                task,
                metadata.get("model_name", "unknown"),
                metadata.get("model_version", "?"),
                model_path.name,
            )
            return bundle

    def availability(self, task: str) -> dict:
        spec = self._specs.get(task)
        if not spec:
            return {"available": False, "reason": "Unknown task."}
        model_path = Path(spec["model_path"])
        metadata_path = Path(spec["metadata_path"])
        if not model_path.exists() or not metadata_path.exists():
            return {
                "available": False,
                "reason": "Model artefact not found. Run the training pipeline.",
            }
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"available": False, "reason": "Model metadata is unreadable."}
        return {
            "available": True,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "trained_at": metadata.get("trained_at"),
            "metadata": metadata,
        }

    def info(self, task: str) -> dict:
        availability = self.availability(task)
        if not availability["available"]:
            return {"task": task, **availability}
        return {"task": task, **availability["metadata"], "available": True}
