"""Evaluate an already-saved SpamGuard AI model without retraining.

This script is deliberately separate from ``training/train.py``:

* it never fits, tunes or changes a model;
* it loads the saved joblib pipeline and the held-out test split;
* it recomputes the same metrics so a report/viva can be reproduced quickly.

Usage (from the project root)::

    python training/evaluate.py --task spam
    python training/evaluate.py --task phishing
    python training/evaluate.py --task all

Results are printed and written to ``reports/evaluation_<task>.json``.
Missing artefacts are reported clearly instead of failing with a stack trace.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

TASKS = {
    "spam": {
        "task_label": "SMS spam",
        "positive": "SPAM",
        "negative": "HAM",
        "model_path": Config.SPAM_MODEL_PATH,
        "metadata_path": Config.SPAM_METADATA_PATH,
        "test_path": Config.PROCESSED_DATA_DIR / "sms_test.csv",
    },
    "phishing": {
        "task_label": "Email phishing",
        "positive": "PHISHING",
        "negative": "LEGITIMATE",
        "model_path": Config.PHISHING_MODEL_PATH,
        "metadata_path": Config.PHISHING_METADATA_PATH,
        "test_path": Config.PROCESSED_DATA_DIR / "phishing_test.csv",
    },
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _evaluate_task(task: str, config: dict) -> dict | None:
    """Evaluate one saved model; return a report dict or None if unavailable."""
    print(f"\n=== {config['task_label'].upper()} ===")
    missing = [
        label
        for label, path in (
            ("model", config["model_path"]),
            ("metadata", config["metadata_path"]),
            ("test split", config["test_path"]),
        )
        if not path.exists()
    ]
    if missing:
        print(
            f"[skip] Missing {', '.join(missing)}. "
            f"Run python training/train.py --task {task} first."
        )
        return None

    metadata = _read_json(config["metadata_path"])
    frame = pd.read_csv(config["test_path"], keep_default_na=False)
    if "message" not in frame.columns or "label_name" not in frame.columns:
        raise ValueError(
            f"{config['test_path']} must contain 'message' and 'label_name' columns."
        )

    pipeline = joblib.load(config["model_path"])
    messages = frame["message"].astype(str).tolist()
    labels = frame["label_name"].astype(str).str.upper().tolist()
    positive = str(metadata.get("positive_class") or config["positive"]).upper()
    y_true = np.asarray([1 if label == positive else 0 for label in labels], dtype=int)

    # Measure inference on the held-out split only. No fitting occurs here.
    started = time.perf_counter()
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        raw = np.asarray(pipeline.predict_proba(messages))
        classes = list(pipeline.classes_)
        if 1 in classes:
            probabilities = raw[:, classes.index(1)]
    threshold = float(metadata.get("classification_threshold", 0.5))
    predictions = (
        (probabilities >= threshold).astype(int)
        if probabilities is not None
        else np.asarray(pipeline.predict(messages), dtype=int)
    )
    elapsed = round(time.perf_counter() - started, 4)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4),
        "roc_auc": (
            round(float(roc_auc_score(y_true, probabilities)), 4)
            if probabilities is not None and len(set(y_true)) == 2
            else None
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "inference_seconds": elapsed,
        "probability_supported": probabilities is not None,
    }
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=[config["negative"], positive],
        output_dict=True,
        zero_division=0,
    )
    serializable_report = {
        key: ({k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in value.items()}
              if isinstance(value, dict) else round(float(value), 4))
        for key, value in report.items()
    }

    result = {
        "task": task,
        "task_label": config["task_label"],
        "selected_model": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "trained_at": metadata.get("trained_at"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(config["model_path"].relative_to(ROOT)),
        "test_path": str(config["test_path"].relative_to(ROOT)),
        "test_rows": int(len(frame)),
        "positive_class": positive,
        "negative_class": config["negative"],
        "decision_threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {
            "labels": [config["negative"], positive],
            "matrix": matrix.tolist(),
        },
        "classification_report": serializable_report,
        "note": (
            "Evaluation of the already-saved pipeline. No model fitting, tuning "
            "or test-set-driven selection occurred in this script."
        ),
    }

    print(f"Model:      {result['selected_model']} v{result['model_version']}")
    print(f"Test rows:  {len(frame):,} ({config['negative']} / {positive})")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1:         {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(
        f"Confusion:  TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,} | "
        f"inference={elapsed:.4f}s"
    )

    output_path = Config.REPORT_DIR / f"evaluation_{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[ok] wrote {output_path.relative_to(ROOT)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate saved SpamGuard AI models (no retraining)."
    )
    parser.add_argument(
        "--task",
        choices=["spam", "phishing", "all"],
        default="all",
        help="Which saved model to evaluate.",
    )
    args = parser.parse_args()

    tasks = ["spam", "phishing"] if args.task == "all" else [args.task]
    results: dict[str, Any] = {}
    missing_explicit = False
    for task in tasks:
        result = _evaluate_task(task, TASKS[task])
        if result is None and args.task != "all":
            missing_explicit = True
        elif result is not None:
            results[task] = result

    if results:
        summary_path = Config.REPORT_DIR / "evaluation_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "models": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[ok] summary written to {summary_path.relative_to(ROOT)}")

    return 1 if missing_explicit else 0


if __name__ == "__main__":
    raise SystemExit(main())
