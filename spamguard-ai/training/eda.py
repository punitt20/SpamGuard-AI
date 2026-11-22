"""Exploratory data analysis for the SMS and phishing datasets.

Usage:
    python training/eda.py

This script only *describes* the data. It does not fit a classifier, select
hyper-parameters, or use the test split for model decisions. Figures and a
machine-readable summary are written to reports/.
"""
from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from training.preprocess import load_phishing, load_sms  # noqa: E402

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0f172a",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "savefig.bbox": "tight",
    }
)

NAVY = "#0f172a"
BLUE = "#3b82f6"
VIOLET = "#8b5cf6"
SPAM = "#ef4444"
HAM = "#10b981"
PHISHING = "#f97316"
LEGIT = "#10b981"
MUTED = "#64748b"

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves", "u", "ur", "im", "dont",
    "get", "got", "will", "want", "know", "like", "one", "also", "may",
    "would", "make", "see", "use", "used", "using", "us", "via", "still",
    "even", "much", "many", "well", "back", "first", "last", "new", "two",
}


def clean_words(text: str) -> list[str]:
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]{1,500}>", " ", text)
    tokens = re.findall(r"[a-z][a-z0-9']+", text)
    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) >= 2 and not token.isdigit()
    ]


def top_tokens(texts, n=20) -> list[tuple[str, int]]:
    counter = Counter()
    for text in texts:
        counter.update(clean_words(text))
    return counter.most_common(n)


def top_bigrams(texts, n=12) -> list[tuple[str, int]]:
    counter = Counter()
    for text in texts:
        words = [word for word in clean_words(text) if word.isalpha()]
        counter.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return counter.most_common(n)


def _source_note(fig, text: str) -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def _save(fig, name: str) -> str:
    path = Config.FIGURE_DIR / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_class_distribution(frame, name, colors, title, source):
    counts = frame["label_name"].value_counts().reindex(
        [label for label in colors], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(counts.index, counts.values, color=[colors[label] for label in counts.index], width=0.58)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.015,
            f"{value:,}\n({value / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY,
        )
    ax.set_title(title)
    ax.set_ylabel("Messages")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _source_note(fig, source)
    return _save(fig, name)


def plot_lengths(frame, name, color, title, source, use_original=False):
    lengths = (
        frame["original_length"] if use_original and "original_length" in frame else frame["message"].str.len()
    )
    cap = int(np.percentile(lengths, 99))
    cap = max(cap, 100)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(np.clip(lengths, 0, cap), bins=50, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(lengths.median(), color=NAVY, linestyle="--", linewidth=1.2, label=f"Median: {lengths.median():.0f}")
    ax.axvline(lengths.mean(), color=VIOLET, linestyle=":", linewidth=1.5, label=f"Mean: {lengths.mean():.0f}")
    ax.set_title(title)
    ax.set_xlabel(f"Characters (clipped at 99th percentile = {cap:,})")
    ax.set_ylabel("Messages")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _source_note(fig, source)
    return _save(fig, name)


def plot_length_by_class(frame, name, colors, title, source):
    labels = list(colors)
    lengths = [frame.loc[frame["label_name"] == label, "message"].str.len() for label in labels]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bp = ax.boxplot(
        lengths,
        tick_labels=labels,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": NAVY, "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 4, "markerfacecolor": MUTED, "markeredgecolor": MUTED, "alpha": 0.5},
    )
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.75)
        patch.set_edgecolor(colors[label])
    ax.set_title(title)
    ax.set_ylabel("Characters")
    ax.set_yscale("symlog")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _source_note(fig, source)
    return _save(fig, name)


def plot_top_words(items, name, color, title, source, xlabel="Frequency"):
    labels = [item[0] for item in reversed(items)]
    values = [item[1] for item in reversed(items)]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.barh(labels, values, color=color, alpha=0.9)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value}", va="center", fontsize=8.5, color=NAVY)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    _source_note(fig, source)
    return _save(fig, name)


def describe(frame, stats, label_colors) -> dict:
    lengths = frame["message"].str.len()
    summary = {
        **stats,
        "rows": int(len(frame)),
        "class_distribution": {
            label: int((frame["label_name"] == label).sum()) for label in label_colors
        },
        "imbalance_ratio": round(
            max((frame["label_name"] == label).sum() for label in label_colors)
            / max(1, min((frame["label_name"] == label).sum() for label in label_colors)),
            2,
        ),
        "length_chars": {
            "mean": round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
            "std": round(float(lengths.std()), 1),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p95": int(np.percentile(lengths, 95)),
        },
        "length_by_class": {
            label: {
                "mean": round(float(frame.loc[frame["label_name"] == label, "message"].str.len().mean()), 1),
                "median": round(float(frame.loc[frame["label_name"] == label, "message"].str.len().median()), 1),
            }
            for label in label_colors
        },
    }
    return summary


def main() -> int:
    Config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sms, sms_stats = load_sms()
    phishing, phish_stats = load_phishing()

    figures = {
        "sms": {},
        "phishing": {},
    }

    sms_colors = {"HAM": HAM, "SPAM": SPAM}
    figures["sms"]["class_distribution"] = plot_class_distribution(
        sms,
        "sms_class_distribution.png",
        sms_colors,
        "SMS Spam Collection: class distribution",
        "Source: UCI SMS Spam Collection v.1 (after duplicate removal)",
    )
    figures["sms"]["message_lengths"] = plot_lengths(
        sms,
        "sms_message_length_distribution.png",
        BLUE,
        "SMS message length distribution",
        "Source: UCI SMS Spam Collection v.1",
    )
    figures["sms"]["length_by_class"] = plot_length_by_class(
        sms,
        "sms_length_by_class.png",
        sms_colors,
        "SMS length by class",
        "Source: UCI SMS Spam Collection v.1",
    )
    sms_spam_words = top_tokens(sms.loc[sms["label_id"] == 1, "message"], 20)
    sms_ham_words = top_tokens(sms.loc[sms["label_id"] == 0, "message"], 20)
    sms_spam_bigrams = top_bigrams(sms.loc[sms["label_id"] == 1, "message"], 12)
    figures["sms"]["top_spam_words"] = plot_top_words(
        sms_spam_words,
        "sms_top_spam_words.png",
        SPAM,
        "Most frequent words in spam SMS",
        "Source: UCI SMS Spam Collection v.1 (stopwords removed)",
    )
    figures["sms"]["top_ham_words"] = plot_top_words(
        sms_ham_words,
        "sms_top_ham_words.png",
        HAM,
        "Most frequent words in legitimate SMS",
        "Source: UCI SMS Spam Collection v.1 (stopwords removed)",
    )
    figures["sms"]["top_spam_bigrams"] = plot_top_words(
        sms_spam_bigrams,
        "sms_top_spam_bigrams.png",
        VIOLET,
        "Most frequent bigrams in spam SMS",
        "Source: UCI SMS Spam Collection v.1",
    )

    phishing_colors = {"LEGITIMATE": LEGIT, "PHISHING": PHISHING}
    figures["phishing"]["class_distribution"] = plot_class_distribution(
        phishing,
        "phishing_class_distribution.png",
        phishing_colors,
        "Phishing email dataset: class distribution",
        "Source: Hugging Face mirror of Kaggle phishing email dataset (deduplicated)",
    )
    figures["phishing"]["message_lengths"] = plot_lengths(
        phishing,
        "phishing_message_length_distribution.png",
        VIOLET,
        "Phishing email length distribution",
        "Source: Hugging Face phishing email dataset (trimmed at 20k chars for modelling)",
        use_original=True,
    )
    figures["phishing"]["length_by_class"] = plot_length_by_class(
        phishing,
        "phishing_length_by_class.png",
        phishing_colors,
        "Phishing email length by class",
        "Source: Hugging Face phishing email dataset",
    )
    phish_words = top_tokens(phishing.loc[phishing["label_id"] == 1, "message"], 20)
    legit_words = top_tokens(phishing.loc[phishing["label_id"] == 0, "message"], 20)
    figures["phishing"]["top_phishing_words"] = plot_top_words(
        phish_words,
        "phishing_top_phishing_words.png",
        PHISHING,
        "Most frequent words in phishing emails",
        "Source: Hugging Face phishing email dataset (stopwords removed)",
    )
    figures["phishing"]["top_legitimate_words"] = plot_top_words(
        legit_words,
        "phishing_top_legitimate_words.png",
        LEGIT,
        "Most frequent words in legitimate emails",
        "Source: Hugging Face phishing email dataset (stopwords removed)",
    )

    sms_summary = describe(sms, sms_stats, sms_colors)
    phish_summary = describe(phishing, phish_stats, phishing_colors)
    split_summary_path = Config.PROCESSED_DATA_DIR / "dataset_summary.json"
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        for summary, key in ((sms_summary, "sms"), (phish_summary, "phishing")):
            source = split_summary.get(key, {})
            for field in (
                "train_rows",
                "test_rows",
                "train_class_counts",
                "test_class_counts",
                "test_size",
                "random_state",
            ):
                summary[field] = source.get(field)
    sms_summary["top_spam_words"] = sms_spam_words
    sms_summary["top_ham_words"] = sms_ham_words
    sms_summary["top_spam_bigrams"] = sms_spam_bigrams
    phish_summary["top_phishing_words"] = phish_words
    phish_summary["top_legitimate_words"] = legit_words
    sms_summary["figures"] = figures["sms"]
    phish_summary["figures"] = figures["phishing"]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Descriptive analysis of the full cleaned corpora. No classifier "
            "fitting or test-set-driven decisions occur here."
        ),
        "sms": sms_summary,
        "phishing": phish_summary,
    }
    (Config.REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    markdown = _markdown(output)
    (Config.REPORT_DIR / "EDA_REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\n[ok] EDA figures saved to {Config.FIGURE_DIR.relative_to(ROOT)}")
    print(f"[ok] summary saved to reports/EDA_REPORT.md")
    return 0


def _markdown(output: dict) -> str:
    sms = output["sms"]
    phish = output["phishing"]
    lines = [
        "# SpamGuard AI — Exploratory Data Analysis",
        "",
        f"Generated: {output['generated_at']}",
        "",
        "> Descriptive only. No classifier fitting or test-set-driven decisions were made in this analysis.",
        "",
        "## Dataset summary",
        "",
        "| Statistic | SMS Spam Collection | Phishing email dataset |",
        "|---|---:|---:|",
        f"| Raw rows | {sms['raw_rows']:,} | {phish['raw_rows']:,} |",
        f"| Missing messages | {sms['missing_messages']:,} | {phish['missing_messages']:,} |",
        f"| Duplicate messages (raw) | {sms['duplicate_messages']:,} | {phish['duplicate_messages']:,} |",
        f"| Clean rows | {sms['clean_rows']:,} | {phish['clean_rows']:,} |",
        f"| Training rows (80%) | {sms['train_rows']:,} | {phish['train_rows']:,} |",
        f"| Test rows (20%) | {sms['test_rows']:,} | {phish['test_rows']:,} |",        f"| Imbalance ratio | {sms['imbalance_ratio']}:1 | {phish['imbalance_ratio']}:1 |",
        f"| Mean length (chars) | {sms['length_chars']['mean']:,} | {phish['length_chars']['mean']:,} |",
        f"| Median length (chars) | {sms['length_chars']['median']:,} | {phish['length_chars']['median']:,} |",
        "",
        "## Class counts",
        "",
        f"- SMS: HAM {sms['class_distribution'].get('HAM', 0):,}; SPAM {sms['class_distribution'].get('SPAM', 0):,}.",
        f"- Phishing: LEGITIMATE {phish['class_distribution'].get('LEGITIMATE', 0):,}; PHISHING {phish['class_distribution'].get('PHISHING', 0):,}.",
        f"- The phishing corpus contains {phish['truncated_messages']:,} emails longer than 20,000 characters; those were truncated for modelling only.",
        "",
        "## SMS: top spam terms",
        "",
        ", ".join(f"{term} ({count})" for term, count in sms["top_spam_words"][:15]),
        "",
        "## SMS: top legitimate terms",
        "",
        ", ".join(f"{term} ({count})" for term, count in sms["top_ham_words"][:15]),
        "",
        "## SMS: top spam bigrams",
        "",
        ", ".join(f"\"{term}\" ({count})" for term, count in sms["top_spam_bigrams"][:10]),
        "",
        "## Phishing: top phishing terms",
        "",
        ", ".join(f"{term} ({count})" for term, count in phish["top_phishing_words"][:15]),
        "",
        "## Phishing: top legitimate terms",
        "",
        ", ".join(f"{term} ({count})" for term, count in phish["top_legitimate_words"][:15]),
        "",
        "## Figures",
        "",
        "All figures are available in `reports/figures/`:",
        "",
    ]
    for task in ("sms", "phishing"):
        lines.append(f"### {task.title()}")
        for label, path in output[task]["figures"].items():
            lines.append(f"- {label.replace('_', ' ').title()}: `{path}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
