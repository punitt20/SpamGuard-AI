"""Dataset download and provenance tracking.

Usage:
    python training/download_dataset.py [--force]

Downloads the UCI SMS Spam Collection and the separable phishing email
dataset. Existing files are left untouched unless --force is passed. No data
is committed to version control; this script lets the project be reproduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
PHISHING_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/"
    "resolve/main/Phishing_Email.csv"
)

SOURCES = {
    "sms": {
        "name": "SMS Spam Collection v.1",
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "doi": "10.24432/C5CC84",
        "license_note": (
            "Distributed by UCI for research use with citation requested; see the "
            "dataset readme for the full custom licence/disclaimer."
        ),
        "citation": (
            "Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). "
            "Contributions to the Study of SMS Spam Filtering: New Collection and Results."
        ),
    },
    "phishing": {
        "name": "Phishing Email Dataset (Hugging Face mirror)",
        "source": PHISHING_URL,
        "original_source": "https://www.kaggle.com/datasets/subhajournal/phishingemails",
        "license_note": "Hugging Face dataset card declares LGPL-3.0; verify the original Kaggle page before redistribution.",
        "citation": "Cyber Cop / subhajournal, 'Phishing Email Detection' dataset (Kaggle).",
        "note": (
            "Direct mirror of a Kaggle dataset created for phishing-email classification. "
            "Labels are 'Safe Email' and 'Phishing Email'. This is an email-domain "
            "dataset and is not mixed with the SMS corpus."
        ),
    },
}


def _download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"[skip] {destination.name} already exists ({destination.stat().st_size:,} bytes)")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}")
    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={"User-Agent": "SpamGuardAI-academic-project/1.0"},
    ) as response:
        response.raise_for_status()
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
        temporary.replace(destination)
    print(f"[ok] {destination.name}: {destination.stat().st_size:,} bytes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SpamGuard AI datasets")
    parser.add_argument("--force", action="store_true", help="replace existing files")
    args = parser.parse_args()

    raw_dir = Config.RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    sms_zip = raw_dir / "sms+spam+collection.zip"
    sms_txt = raw_dir / "SMSSpamCollection"
    phishing_csv = raw_dir / "Phishing_Email.csv"

    _download(SMS_URL, sms_zip, args.force)
    if not sms_txt.exists() or args.force:
        with zipfile.ZipFile(sms_zip) as archive:
            archive.extract("SMSSpamCollection", raw_dir)
            if (raw_dir / "readme").exists():
                pass
            archive.extract("readme", raw_dir)
        print(f"[ok] extracted {sms_txt.name}")

    _download(PHISHING_URL, phishing_csv, args.force)

    sources = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {},
    }
    for key, path in (("sms", sms_txt), ("phishing", phishing_csv)):
        sources["datasets"][key] = {
            **SOURCES[key],
            "file": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    provenance = raw_dir / "SOURCES.json"
    provenance.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"[ok] provenance written to {provenance.relative_to(ROOT)}")
    print("\nNext:  python training/preprocess.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
