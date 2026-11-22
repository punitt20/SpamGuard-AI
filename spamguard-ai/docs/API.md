# API reference

Local development base URL: `http://127.0.0.1:5001`.

## `POST /api/predict`

Classifies one SMS using the available spam model.

```json
{
  "message": "Congratulations! You have won a FREE prize. Claim now.",
  "task": "spam"
}
```

`message` must be a non-empty string of at most 20,000 characters. `task` is
optional and defaults to `spam`; no other task is currently deployed.

A successful response includes the cleaned message, task, `classification`
(`SPAM` or `HAM`), `model_probability` (the spam-class probability), model
name and version, the `decision_threshold` used for the verdict, and
`top_signals` when the model can provide them. Signal
weights show model associations and are not causal explanations.

Errors return JSON with `error`, `code`, and `status` fields:

| Status | Code | Meaning |
|---|---|---|
| 400 | `invalid_json` | Body is missing, malformed, or not a JSON object |
| 400 | `invalid_message` | Message is missing, not a string, empty, too long, or task is unsupported |
| 503 | `model_unavailable` | Model artifact is missing, unloadable, or unable to process input |

## `GET /api/model`

Returns model availability and metadata for the configured spam and phishing
tasks. The phishing model is currently unavailable unless an artifact has been
trained and configured.

## `GET /api/model/<task>`

Returns detailed evaluation metadata for `spam` or `phishing`. An unavailable
model returns 404.

## `GET /api/analytics?task=all|spam|phishing`

Returns saved model comparison data only. It does not track live message
counts or message history.
