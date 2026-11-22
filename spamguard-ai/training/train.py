"""Train, tune, evaluate, and persist the SpamGuard AI classifiers.

Usage:
    python training/train.py                 # train spam + phishing models
    python training/train.py --task spam
    python training/train.py --task phishing

Hyper-parameter tuning uses stratified cross-validation on the training split
only. The held-out test split is touched once, after fitting, for the reported
comparison metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from app.services.preprocessing import (  # noqa: E402
    PREPROCESSING_VERSION,
    TextNormalizer,
)

RANDOM_STATE = 42
MODEL_VERSION = "1.1.0"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0f172a",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "grid.color": "#e2e8f0",
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
    }
)

BLUE = "#3b82f6"
VIOLET = "#8b5cf6"
NAVY = "#0f172a"
SPAM = "#ef4444"
HAM = "#10b981"
PHISHING = "#f97316"
LEGIT = "#10b981"
MUTED = "#64748b"

TASKS = {
    "spam": {
        "prefix": "sms",
        "dataset_name": "SMS Spam Collection v.1",
        "dataset_source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "doi": "10.24432/C5CC84",
        "citation": (
            "Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). "
            "Contributions to the Study of SMS Spam Filtering."
        ),
        "positive": "SPAM",
        "negative": "HAM",
        "task_label": "SMS spam detection",
        "input_domain": "SMS and short text messages",
        "model_path": Config.SPAM_MODEL_PATH,
        "metadata_path": Config.SPAM_METADATA_PATH,
        "comparison_path": Config.REPORT_DIR / "model_comparison.csv",
        "cv_folds": 5,
        "n_jobs": -1,
        "limitations": [
            "Trained on English SMS messages (mostly UK/Singapore sources).",
            "Class imbalance means minority-class errors deserve attention despite high accuracy.",
            "The model detects statistical spam patterns; it cannot verify sender identity or links.",
            "Probability estimates are calibrated only where the selected algorithm supports it.",
            "Keyword/URL heuristics are not used by the classifier and do not override it.",
        ],
    },
    "phishing": {
        "prefix": "phishing",
        "dataset_name": "Phishing Email Dataset (Hugging Face mirror)",
        "dataset_source": "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset",
        "doi": None,
        "citation": (
            "Cyber Cop / subhajournal, 'Phishing Email Detection' dataset (Kaggle); "
            "Hugging Face mirror under LGPL-3.0."
        ),
        "positive": "PHISHING",
        "negative": "LEGITIMATE",
        "task_label": "Email phishing detection",
        "input_domain": "Email text (not SMS)",
        "model_path": Config.PHISHING_MODEL_PATH,
        "metadata_path": Config.PHISHING_METADATA_PATH,
        "comparison_path": Config.REPORT_DIR / "phishing_model_comparison.csv",
        "cv_folds": 3,
        "n_jobs": 2,
        "limitations": [
            "Trained on an email corpus, not SMS; results should not be transferred to SMS without validation.",
            "The dataset is a Kaggle/Hugging Face mirror with documented but third-party provenance.",
            "Very long emails are truncated at 20,000 characters for modelling.",
            "The classifier assesses text patterns; it does not inspect headers, attachments, or live URLs.",
            "Phishing predictions are kept separate from SMS spam predictions and are not merged.",
        ],
    },
}

CANDIDATE_STYLES = {
    "Multinomial Naive Bayes": BLUE,
    "Logistic Regression": VIOLET,
    "Linear SVM": MUTED,
    "Linear SVM (calibrated)": SPAM,
}


def _load_split(task: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = TASKS[task]
    processed = Config.PROCESSED_DATA_DIR
    train_path = processed / f"{cfg['prefix']}_train.csv"
    test_path = processed / f"{cfg['prefix']}_test.csv"
    summary_path = processed / "dataset_summary.json"
    if not train_path.exists() or not test_path.exists() or not summary_path.exists():
        print(f"[setup] processed data missing; preparing datasets...")
        from training.preprocess import prepare

        prepare(force=True)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))[cfg["prefix"]]

    required = {"message", "label_id"}
    if not required.issubset(train.columns) or not required.issubset(test.columns):
        raise ValueError(f"Processed {task} files are missing required columns.")
    if train["message"].isna().any() or test["message"].isna().any():
        raise ValueError("Missing messages found in processed split.")
    overlap = set(train["message"]).intersection(set(test["message"]))
    if overlap:
        raise ValueError(f"Data leakage detected: {len(overlap)} messages in both splits.")
    for label in (0, 1):
        if label not in train["label_id"].values or label not in test["label_id"].values:
            raise ValueError(f"Class {label} missing from a {task} split.")
    return train, test, summary


def build_pipeline(classifier) -> Pipeline:
    """TF-IDF pipeline; fitted on training data only."""
    return Pipeline(
        steps=[
            ("normalizer", TextNormalizer(remove_stopwords=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("clf", classifier),
        ]
    )


def evaluate_pipeline(pipeline, X_test, y_test, task_cfg, threshold: float | None = None) -> dict:
    started = time.perf_counter()
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            positive_index = list(pipeline.classes_).index(1)
            probabilities = np.asarray(pipeline.predict_proba(X_test))[:, positive_index]
        except Exception:
            probabilities = None
    y_pred = (
        (probabilities >= threshold).astype(int)
        if threshold is not None and probabilities is not None
        else np.asarray(pipeline.predict(X_test))
    )
    inference_seconds = time.perf_counter() - started
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[task_cfg["negative"], task_cfg["positive"]],
        output_dict=True,
        zero_division=0,
    )
    probability_supported = hasattr(pipeline, "predict_proba")
    roc_auc = None
    if probabilities is not None:
        roc_auc = float(roc_auc_score(y_test, probabilities))

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "roc_auc": None if roc_auc is None else round(roc_auc, 4),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "inference_seconds": round(inference_seconds, 4),
        "probability_supported": probability_supported,
        "decision_threshold": threshold,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "_probabilities": probabilities,
    }


def _clean_best_params(best_params: dict) -> dict:
    return {
        key.replace("clf__", ""): value for key, value in best_params.items()
    }


def _fit_candidate(
    display_name: str,
    estimator,
    grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    task_cfg: dict,
    probability_supported: bool,
    notes: str = "",
) -> dict:
    pipeline = build_pipeline(estimator)
    cv = StratifiedKFold(
        n_splits=task_cfg["cv_folds"], shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="f1",
        cv=cv,
        n_jobs=task_cfg.get("n_jobs", -1),
        refit=True,
        return_train_score=False,
    )
    print(f"[train] {display_name}: {len(_param_combinations(grid))} settings × {task_cfg['cv_folds']} folds")
    started = time.perf_counter()
    search.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started
    threshold = None
    cv_f1_selected_threshold = float(search.best_score_)
    cv_precision_selected_threshold = None
    cv_recall_selected_threshold = None
    if probability_supported:
        threshold, cv_f1_selected_threshold, cv_precision_selected_threshold, cv_recall_selected_threshold = _cross_validated_threshold(
            search.best_estimator_, X_train, y_train, task_cfg
        )
    metrics = evaluate_pipeline(search.best_estimator_, X_test, y_test, task_cfg, threshold)
    best_index = search.best_index_
    cv_std = float(search.cv_results_["std_test_score"][best_index])
    print(
        f"        best F1={metrics['f1_score']:.4f} | P={metrics['precision']:.4f} "
        f"| R={metrics['recall']:.4f} | accuracy={metrics['accuracy']:.4f} "
        f"| time={training_seconds:.1f}s"
    )
    return {
        "model_name": display_name,
        "probability_supported": probability_supported,
        "pipeline": search.best_estimator_,
        "metrics": metrics,
        "best_params": _clean_best_params(search.best_params_),
        "cv_f1_mean": round(float(search.best_score_), 4),
        "cv_f1_std": round(cv_std, 4),
        "cv_f1_selected_threshold": round(cv_f1_selected_threshold, 4),
        "cv_precision_selected_threshold": (
            None if cv_precision_selected_threshold is None else round(cv_precision_selected_threshold, 4)
        ),
        "cv_recall_selected_threshold": (
            None if cv_recall_selected_threshold is None else round(cv_recall_selected_threshold, 4)
        ),
        "decision_threshold": threshold,
        "training_time_seconds": round(training_seconds, 2),
        "notes": notes,
    }


def _cross_validated_threshold(pipeline, X_train, y_train, task_cfg):
    """Choose the probability cutoff from out-of-fold predictions on training data."""
    cv = StratifiedKFold(
        n_splits=task_cfg["cv_folds"], shuffle=True, random_state=RANDOM_STATE
    )
    fold_probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=task_cfg.get("n_jobs", -1),
    )
    positive_index = list(pipeline.classes_).index(1)
    positive_probability = fold_probabilities[:, positive_index]

    best = None
    for threshold in np.arange(0.10, 0.91, 0.05):
        prediction = (positive_probability >= threshold).astype(int)
        score = (
            float(f1_score(y_train, prediction, zero_division=0)),
            float(precision_score(y_train, prediction, zero_division=0)),
            -abs(float(threshold) - 0.5),
        )
        if best is None or score > best[0]:
            best = (score, float(threshold), prediction)
    score, threshold, prediction = best
    return (
        round(threshold, 2),
        score[0],
        float(precision_score(y_train, prediction, zero_division=0)),
        float(recall_score(y_train, prediction, zero_division=0)),
    )


def _param_combinations(grid: dict) -> list[dict]:
    keys = list(grid)
    combos = [{}]
    for key in keys:
        values = grid[key]
        combos = [dict(item, **{key: value}) for item in combos for value in values]
    return combos


def train_task(task: str) -> dict:
    cfg = TASKS[task]
    train, test, data_summary = _load_split(task)
    X_train, y_train = train["message"].tolist(), train["label_id"].astype(int).tolist()
    X_test, y_test = test["message"].tolist(), test["label_id"].astype(int).tolist()
    print(
        f"\n=== {task.upper()} ===\ntrain={len(train):,} test={len(test):,} "
        f"classes={train['label_name'].value_counts().to_dict()}"
    )

    candidates: list[dict] = []

    candidates.append(
        _fit_candidate(
            "Multinomial Naive Bayes",
            MultinomialNB(),
            {"clf__alpha": [0.1, 0.5, 1.0]},
            X_train,
            y_train,
            X_test,
            y_test,
            cfg,
            probability_supported=True,
            notes="Fast probabilistic baseline; alpha tuned by training-only CV.",
        )
    )
    candidates.append(
        _fit_candidate(
            "Logistic Regression",
            LogisticRegression(solver="liblinear", max_iter=3000, random_state=RANDOM_STATE),
            {"clf__C": [0.5, 2.0, 10.0], "clf__class_weight": [None, "balanced"]},
            X_train,
            y_train,
            X_test,
            y_test,
            cfg,
            probability_supported=True,
            notes="L2-regularised linear model with native probability estimates.",
        )
    )
    svm_candidate = _fit_candidate(
        "Linear SVM",
        LinearSVC(dual="auto", max_iter=8000, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1.0], "clf__class_weight": [None, "balanced"]},
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        probability_supported=False,
        notes=(
            "Decision-function margin only; raw scores are not probabilities "
            "and are excluded from the probability-based selection."
        ),
    )
    candidates.append(svm_candidate)

    # A calibrated variant makes the SVM comparable for probability-aware use.
    svm_params = {
        key: value
        for key, value in svm_candidate["best_params"].items()
        if key in {"C", "class_weight"}
    }
    calibrated_base = LinearSVC(
        dual="auto", max_iter=8000, random_state=RANDOM_STATE, **svm_params
    )
    calibrated = CalibratedClassifierCV(
        calibrated_base,
        method="sigmoid",
        cv=cfg["cv_folds"],
    )
    print(f"[train] Linear SVM (calibrated): sigmoid calibration, cv={cfg['cv_folds']}")
    started = time.perf_counter()
    calibrated_pipeline = build_pipeline(calibrated)
    calibrated_pipeline.fit(X_train, y_train)
    calibration_seconds = time.perf_counter() - started
    cal_threshold, cal_cv_f1, cal_cv_precision, cal_cv_recall = _cross_validated_threshold(
        calibrated_pipeline, X_train, y_train, cfg
    )
    cal_metrics = evaluate_pipeline(calibrated_pipeline, X_test, y_test, cfg, cal_threshold)
    print(
        f"        best F1={cal_metrics['f1_score']:.4f} | P={cal_metrics['precision']:.4f} "
        f"| R={cal_metrics['recall']:.4f} | accuracy={cal_metrics['accuracy']:.4f} "
        f"| time={calibration_seconds:.1f}s"
    )
    candidates.append(
        {
            "model_name": "Linear SVM (calibrated)",
            "probability_supported": True,
            "pipeline": calibrated_pipeline,
            "metrics": cal_metrics,
            "best_params": svm_params,
            "cv_f1_mean": None,
            "cv_f1_std": None,
            "cv_f1_selected_threshold": round(cal_cv_f1, 4),
            "cv_precision_selected_threshold": round(cal_cv_precision, 4),
            "cv_recall_selected_threshold": round(cal_cv_recall, 4),
            "decision_threshold": cal_threshold,
            "training_time_seconds": round(calibration_seconds, 2),
            "notes": (
                "Platt-sigmoid calibration of the tuned Linear SVM; calibration "
                "uses the training split only. Training time excludes the SVM grid search."
            ),
        }
    )

    probability_candidates = [
        candidate for candidate in candidates if candidate["probability_supported"]
    ]
    selected = max(
        probability_candidates,
        key=lambda candidate: (
            candidate["cv_f1_selected_threshold"],
            candidate["cv_precision_selected_threshold"] or 0,
            -abs((candidate["decision_threshold"] or 0.5) - 0.5),
        ),
    )
    selected_name = selected["model_name"]

    comparison = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        comparison.append(
            {
                "model_name": candidate["model_name"],
                "probability_supported": candidate["probability_supported"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc": metrics["roc_auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "cv_f1_mean": candidate["cv_f1_mean"],
                "cv_f1_std": candidate["cv_f1_std"],
                "cv_f1_selected_threshold": candidate["cv_f1_selected_threshold"],
                "cv_precision_selected_threshold": candidate["cv_precision_selected_threshold"],
                "cv_recall_selected_threshold": candidate["cv_recall_selected_threshold"],
                "decision_threshold": candidate["decision_threshold"],
                "training_time_seconds": candidate["training_time_seconds"],
                "inference_seconds": metrics["inference_seconds"],
                "best_params": candidate["best_params"],
                "is_selected": candidate["model_name"] == selected_name,
                "notes": candidate["notes"],
            }
        )

    comparison_frame = pd.DataFrame(
        [
            {
                **{k: v for k, v in row.items() if k not in {"best_params", "notes"}},
                "best_params": json.dumps(row["best_params"]),
                "notes": row["notes"],
            }
            for row in comparison
        ]
    )
    Config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame.to_csv(cfg["comparison_path"], index=False)

    selected_metrics = selected["metrics"]
    rationale = (
        f"Selected {selected_name} by its highest training-only out-of-fold F1 "
        f"({selected['cv_f1_selected_threshold']:.4f}) at a probability threshold of "
        f"{selected['decision_threshold']:.2f}; precision was used as a tie-breaker. "
        "The held-out test split was not used for model or threshold selection and is "
        "reserved for the final reported metrics. Only probability-capable models are eligible."
    )
    figure_paths = generate_figures(task, cfg, candidates, selected, y_test)

    metadata = {
        "task": task,
        "task_label": cfg["task_label"],
        "input_domain": cfg["input_domain"],
        "model_name": selected_name,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "probability_supported": True,
        "selection_criterion": (
            "Training-only out-of-fold F1 for the positive class after selecting a probability "
            "threshold from 0.10 to 0.90; precision is the tie-breaker. The held-out test split "
            "is reserved for final reporting. Probability-capable models only."
        ),
        "selection_rationale": rationale,
        "class_labels": {"0": cfg["negative"], "1": cfg["positive"]},
        "positive_class": cfg["positive"],
        "dataset": {
            "name": cfg["dataset_name"],
            "source": cfg["dataset_source"],
            "license_note": (
                "UCI SMS Spam Collection: research use with citation; see dataset readme."
                if task == "spam"
                else "Hugging Face dataset card declares LGPL-3.0; original Kaggle source."
            ),
            "doi": cfg["doi"],
            "citation": cfg["citation"],
            "raw_rows": data_summary.get("raw_rows"),
            "clean_rows": data_summary.get("clean_rows"),
            "duplicates_removed": data_summary.get("duplicates_removed"),
            "missing_messages": data_summary.get("missing_messages"),
            "conflicting_label_messages": data_summary.get("conflicting_label_messages"),
            "truncated_messages": data_summary.get("truncated_messages", 0),
            "max_text_chars": data_summary.get("max_text_chars", 20_000),
            "raw_sha256": data_summary.get("raw_sha256"),
            "train_rows": data_summary.get("train_rows"),
            "test_rows": data_summary.get("test_rows"),
            "train_class_counts": data_summary.get("train_class_counts"),
            "test_class_counts": data_summary.get("test_class_counts"),
        },
        "split": {
            "test_size": data_summary.get("test_size", 0.2),
            "random_state": data_summary.get("random_state", RANDOM_STATE),
            "stratified": True,
        },
        "features": {
            "method": "TF-IDF",
            "max_features": 5000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "token_pattern": r"(?u)\b\w+\b",
            "fit_on": "training split only",
            "rationale": (
                "min_df=2 removes terms seen once in the training corpus, limiting "
                "vocabulary noise; max_df=0.95 suppresses near-universal terms. Bigrams "
                "capture phrases such as 'free prize' while unigrams preserve recall."
            ),
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "remove_stopwords": False,
            "note": (
                "Lower-casing, HTML/entity handling, URL placeholder + hostname tokens, "
                "e-mail placeholder, currency token, punctuation/whitespace normalisation, "
                "and signal flags for excessive exclamation/uppercase. Stopword removal "
                "is available but disabled so urgency words such as 'now' are retained."
            ),
        },
        "pipeline_steps": [
            "TextNormalizer (custom, shared with inference)",
            "TfidfVectorizer (max_features=5000, ngram_range=(1,2))",
            f"Classifier: {type(selected['pipeline'].named_steps['clf']).__name__}",
        ],
        "metrics": {
            key: value
            for key, value in selected_metrics.items()
            if key not in {"_probabilities", "classification_report", "confusion_matrix"}
        },
        "classification_report": selected_metrics["classification_report"],
        "confusion_matrix": {
            "labels": [cfg["negative"], cfg["positive"]],
            "matrix": selected_metrics["confusion_matrix"],
        },
        "best_params": selected["best_params"],
        "classification_threshold": selected["decision_threshold"] or 0.5,
        "cv_f1_mean": selected["cv_f1_mean"],
        "cv_f1_std": selected["cv_f1_std"],
        "comparison": comparison,
        "figures": figure_paths,
        "artifacts": {
            "model_path": str(cfg["model_path"].relative_to(ROOT)),
            "metadata_path": str(cfg["metadata_path"].relative_to(ROOT)),
            "comparison_csv": str(cfg["comparison_path"].relative_to(ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "training_command": f"python training/train.py --task {task}",
        "limitations": cfg["limitations"],
    }

    # Serialise only after all data needed for metadata has been collected.
    cfg["model_path"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected["pipeline"], cfg["model_path"], compress=3)
    model_bytes = cfg["model_path"].stat().st_size
    metadata["artifacts"]["model_sha256"] = _sha256(cfg["model_path"])
    metadata["artifacts"]["model_size_bytes"] = model_bytes
    cfg["metadata_path"].write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    (Config.REPORT_DIR / f"evaluation_{task}.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "metrics": metadata["metrics"],
                "classification_report": selected_metrics["classification_report"],
                "confusion_matrix": metadata["confusion_matrix"],
                "comparison": [
                    {key: value for key, value in row.items() if key != "best_params"}
                    for row in comparison
                ],
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    print(f"[save] {cfg['model_path']} ({model_bytes / 1024:.1f} KB)")
    print(f"[save] {cfg['metadata_path']}")
    print(f"[save] {cfg['comparison_path']}")
    print(
        f"\nSelected {selected_name} for {task}: "
        f"F1={selected_metrics['f1_score']:.4f}, "
        f"precision={selected_metrics['precision']:.4f}, "
        f"recall={selected_metrics['recall']:.4f}, "
        f"accuracy={selected_metrics['accuracy']:.4f}"
    )
    return metadata


def generate_figures(
    task: str, cfg: dict, candidates: list[dict], selected: dict, y_test
) -> dict:
    """Create comparison, confusion-matrix and ROC figures from real test results."""
    Config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    names = [candidate["model_name"] for candidate in candidates]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(names))
    width = 0.18
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        values = [candidate["metrics"][metric] for candidate in candidates]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color=NAVY,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Score on held-out test set")
    ax.set_title(f"{cfg['task_label']}: candidate model comparison")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    comparison_path = Config.FIGURE_DIR / f"model_comparison_{task}.png"
    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)

    # Confusion matrices for every candidate
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    for ax, candidate in zip(axes.ravel(), candidates):
        cm = np.asarray(candidate["metrics"]["confusion_matrix"])
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() or 1)
        for row in range(2):
            for col in range(2):
                value = cm[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if value > cm.max() * 0.55 else NAVY,
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([cfg["negative"], cfg["positive"]], fontsize=8)
        ax.set_yticklabels([cfg["negative"], cfg["positive"]], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(
            f"{candidate['model_name']}\nF1={candidate['metrics']['f1_score']:.3f}",
            fontsize=10,
        )
    fig.suptitle(
        f"{cfg['task_label']}: confusion matrices (held-out test set)",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(image, ax=axes, shrink=0.6, label="Messages")
    confusion_path = Config.FIGURE_DIR / f"confusion_matrices_{task}.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    # ROC curves for candidates exposing predict_proba
    roc_path = None
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    plotted = 0
    for candidate in candidates:
        probabilities = candidate["metrics"].get("_probabilities")
        if probabilities is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = candidate["metrics"].get("roc_auc")
        ax.plot(
            fpr,
            tpr,
            linewidth=1.8,
            label=f"{candidate['model_name']} (AUC {auc:.3f})",
            color=CANDIDATE_STYLES.get(candidate["model_name"], BLUE),
        )
        plotted += 1
    if plotted:
        ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{cfg['task_label']}: ROC curves (held-out test set)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(alpha=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        roc_path = Config.FIGURE_DIR / f"roc_curves_{task}.png"
        fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    return {
        "model_comparison": str(comparison_path.relative_to(ROOT)),
        "confusion_matrices": str(confusion_path.relative_to(ROOT)),
        "roc_curves": str(roc_path.relative_to(ROOT)) if roc_path else None,
        "selected_model": selected["model_name"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SpamGuard AI models")
    parser.add_argument("--task", choices=["spam", "phishing", "all"], default="all")
    args = parser.parse_args()
    tasks = ["spam", "phishing"] if args.task == "all" else [args.task]
    for task in tasks:
        train_task(task)
    print("\n[ok] Training complete. Start the app with: python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
