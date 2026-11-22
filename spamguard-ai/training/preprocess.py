"""Clean datasets and create stratified train/test splits.

Usage:
    python training/preprocess.py

Outputs (data/processed/):
    sms_clean.csv, sms_train.csv, sms_test.csv
    phishing_clean.csv, phishing_train.csv, phishing_test.csv
    dataset_summary.json

The test split is written to disk once and is not used by any fitting or
hyper-parameter selection code. All model tuning uses cross-validation on the
training split only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_TEXT_CHARS = Config.MODEL_MAX_TEXT_CHARS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sms(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = path or Config.RAW_DATA_DIR / "SMSSpamCollection"
    if not path.exists():
        raise FileNotFoundError(
            f"SMS dataset not found at {path}. Run training/download_dataset.py first."
        )
    raw = pd.read_csv(
        path, sep="\t", header=None, names=["label", "message"], encoding="utf-8"
    )
    raw.insert(0, "source_index", range(len(raw)))
    stats = {
        "raw_rows": int(len(raw)),
        "missing_messages": int(raw["message"].isna().sum()),
        "duplicate_messages": int(raw.duplicated(subset=["message"]).sum()),
        "raw_class_counts": {str(k): int(v) for k, v in raw["label"].value_counts().items()},
    }

    frame = raw.dropna(subset=["message", "label"]).copy()
    frame["label"] = frame["label"].str.strip().str.lower()
    invalid_labels = sorted(set(frame["label"]) - {"ham", "spam"})
    if invalid_labels:
        raise ValueError(f"Unexpected SMS labels: {invalid_labels}")

    # Normalise surrounding whitespace before deduplication; otherwise variants
    # can become identical only after splitting and leak across train/test.
    frame["message"] = frame["message"].astype(str).str.strip()
    frame = frame[frame["message"].str.len() > 0]

    # Identical messages with conflicting labels would teach contradictory
    # targets. There are none in this corpus, but the guard is deliberate.
    conflicts = frame.groupby("message")["label"].nunique()
    duplicate_conflicts = frame[frame["message"].isin(conflicts[conflicts > 1].index)]
    if len(duplicate_conflicts):
        frame = frame[~frame["message"].isin(duplicate_conflicts["message"])]
    frame = frame.drop_duplicates(subset=["message"], keep="first")

    frame["label_id"] = frame["label"].map({"ham": 0, "spam": 1}).astype(int)
    frame["label_name"] = frame["label_id"].map({0: "HAM", 1: "SPAM"})
    frame = frame.reset_index(drop=True)

    stats.update(
        {
            "clean_rows": int(len(frame)),
            "duplicates_removed": stats["raw_rows"] - int(len(frame)),
            "class_counts": {
                "HAM": int((frame["label_id"] == 0).sum()),
                "SPAM": int((frame["label_id"] == 1).sum()),
            },
            "conflicting_label_messages": int(len(duplicate_conflicts)),
        }
    )
    return frame[["source_index", "message", "label_id", "label_name"]], stats


def load_phishing(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    path = path or Config.RAW_DATA_DIR / "Phishing_Email.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Phishing dataset not found at {path}. Run training/download_dataset.py first."
        )
    raw = pd.read_csv(path)
    if "Email Text" not in raw.columns or "Email Type" not in raw.columns:
        raise ValueError("Phishing CSV must contain 'Email Text' and 'Email Type' columns.")
    raw = raw.rename(columns={"Email Text": "message", "Email Type": "label"})
    raw.insert(0, "source_index", range(len(raw)))
    stats = {
        "raw_rows": int(len(raw)),
        "missing_messages": int(raw["message"].isna().sum()),
        "duplicate_messages": int(raw.duplicated(subset=["message"]).sum()),
        "raw_class_counts": {str(k): int(v) for k, v in raw["label"].value_counts().items()},
    }

    frame = raw.dropna(subset=["message", "label"]).copy()
    frame["message"] = frame["message"].astype(str).str.strip()
    frame["label"] = frame["label"].str.strip().str.lower()
    label_map = {"safe email": 0, "phishing email": 1}
    unexpected = sorted(set(frame["label"]) - set(label_map))
    if unexpected:
        raise ValueError(f"Unexpected phishing labels: {unexpected}")
    frame["label_id"] = frame["label"].map(label_map).astype(int)
    frame = frame[frame["message"].str.len() > 0]

    conflicts = frame.groupby("message")["label_id"].nunique()
    duplicate_conflicts = frame[frame["message"].isin(conflicts[conflicts > 1].index)]
    if len(duplicate_conflicts):
        frame = frame[~frame["message"].isin(duplicate_conflicts["message"])]
    frame = frame.drop_duplicates(subset=["message"], keep="first")
    frame["label_name"] = frame["label_id"].map({0: "LEGITIMATE", 1: "PHISHING"})

    # Very long emails are truncated for modelling only. This is recorded so
    # the limitation is visible and reproducible. The raw CSV is unchanged.
    truncated = int((frame["message"].str.len() > MAX_TEXT_CHARS).sum())
    frame["original_length"] = frame["message"].str.len().astype(int)
    frame["message"] = frame["message"].str.slice(0, MAX_TEXT_CHARS)
    frame = frame.reset_index(drop=True)

    stats.update(
        {
            "clean_rows": int(len(frame)),
            "duplicates_removed": stats["raw_rows"] - int(len(frame)),
            "conflicting_label_messages": int(len(duplicate_conflicts)),
            "truncated_messages": truncated,
            "max_text_chars": MAX_TEXT_CHARS,
            "class_counts": {
                "LEGITIMATE": int((frame["label_id"] == 0).sum()),
                "PHISHING": int((frame["label_id"] == 1).sum()),
            },
        }
    )
    return frame[
        ["source_index", "message", "label_id", "label_name", "original_length"]
    ], stats


def _split_and_save(
    frame: pd.DataFrame, prefix: str, processed_dir: Path
) -> dict:
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["label_id"],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(processed_dir / f"{prefix}_train.csv", index=False)
    test.to_csv(processed_dir / f"{prefix}_test.csv", index=False)
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_class_counts": {
            str(k): int(v) for k, v in train["label_name"].value_counts().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in test["label_name"].value_counts().items()
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }


def prepare(force: bool = False) -> dict:
    processed_dir = Config.PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = processed_dir / "dataset_summary.json"
    if summary_path.exists() and not force:
        # The trainer can reuse deterministic artefacts safely; re-running with
        # --force recreates them from raw data.
        pass

    sms, sms_stats = load_sms()
    phishing, phishing_stats = load_phishing()
    sms.to_csv(processed_dir / "sms_clean.csv", index=False)
    phishing.to_csv(processed_dir / "phishing_clean.csv", index=False)
    sms_split = _split_and_save(sms, "sms", processed_dir)
    phishing_split = _split_and_save(phishing, "phishing", processed_dir)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preprocessing_version": "1.0.0",
        "sms": {
            **sms_stats,
            **sms_split,
            "dataset": "SMS Spam Collection v.1",
            "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
            "raw_sha256": _sha256(Config.RAW_DATA_DIR / "SMSSpamCollection"),
        },
        "phishing": {
            **phishing_stats,
            **phishing_split,
            "dataset": "Phishing Email Dataset (Hugging Face mirror)",
            "source": "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset",
            "raw_sha256": _sha256(Config.RAW_DATA_DIR / "Phishing_Email.csv"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[ok] processed data written to {processed_dir}")
    return summary


def main() -> int:
    prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
