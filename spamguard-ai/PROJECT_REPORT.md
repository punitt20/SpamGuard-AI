# SpamGuard AI — Project Summary

SpamGuard AI is a local Flask application for SMS spam classification. It uses
a saved TF-IDF pipeline and exposes a single-message prediction API, model
information, and read-only training evaluation comparisons.

## Runtime

- `POST /api/predict` accepts a non-empty message and returns SPAM or HAM with
  the spam-class probability and model feature signals.
- `GET /api/model` reports configured model availability and metadata.
- `GET /api/analytics` returns candidate evaluation data from saved metadata.
- Messages are processed per request and are not persisted. There is no
  database, analysis history, or usage analytics.

## Model

The current SMS pipeline is stored in `models/spam_classifier.pkl`. See
`models/model_metadata.json` for its dataset, held-out metrics, limitations,
and candidate comparison. The deployed model is a calibrated linear SVM with a
0.40 spam cutoff, selected using training-only out-of-fold F1. On the saved
held-out split, spam recall is 0.9297, precision is 0.9370, and F1 is 0.9333.
The held-out split is reserved for final reporting.

## Run and verify

```bash
.venv/bin/python run.py
.venv/bin/python -m pytest
```
