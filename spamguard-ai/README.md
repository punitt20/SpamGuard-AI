# SpamGuard AI

SpamGuard AI is a local Flask app for analyzing SMS messages with a trained
TF-IDF classifier. It includes an analyzer, model details, and a model
evaluation comparison page. The trained SMS model is included in `models/`.

## Run locally

Use Python 3.12, which matches the included model artifact's training
environment. From the project directory, create an environment and install the
dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then start the app:

```bash
python run.py
```

Open <http://127.0.0.1:5001>. The app binds to localhost by default. The
analyzer sends the message to this local Flask process for inference; messages
are not retained in analysis history. Bootstrap and icon assets are served
from the app, so the interface does not need a CDN connection.

## Pages

- `/analyzer` — classify an SMS as SPAM or HAM and inspect the model's spam
  probability and token signals.
- `/model` — view model availability, metadata, and candidate comparisons.
- `/analytics` — view saved held-out evaluation comparisons. This is not a
  live usage dashboard because analysis history is not persisted.

Only SMS spam prediction is enabled in the app. The optional phishing dataset
and trainer are separate offline experiments.

## API

`POST /api/predict` accepts a JSON object with a non-empty `message` string
(maximum 20,000 characters) and an optional `task` set to `spam`.

```bash
curl -s http://127.0.0.1:5001/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"message":"Congratulations! You have won a FREE prize. Claim now."}'
```

Other read-only endpoints provide saved model information (`/api/model`) and
evaluation summaries (`/api/analytics`). See [`docs/API.md`](docs/API.md) for
the request and response details.

## Model and limitations

The model was selected using training-only cross-validation and a spam
probability cutoff selected from out-of-fold predictions. Its final metrics
are measured on the held-out test split. The model was trained on English SMS
data and produces a statistical signal, not proof that a message or sender is
safe or fraudulent. Its probability is
the model's positive-class score and may be wrong, especially for messages
that differ from its training data. See `models/model_metadata.json` for
evaluation metrics and dataset notes.

## Tests

```bash
.venv/bin/python -m pytest
```

## Retraining

The committed SMS model is ready to run without downloading training data.
To recreate training datasets and retrain, run `python training/download_dataset.py`,
then `python training/preprocess.py` and `python training/train.py --task spam`.
Dataset downloads are optional and may be large; the phishing corpus is kept
out of version control because it is not needed by the deployed SMS analyzer.
