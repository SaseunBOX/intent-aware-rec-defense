"""Reusable preprocessing helpers extracted from `scripts/preprocess_all_e.py`.

This module intentionally preserves the prototype's current path layout and
heuristic logic. It does not run the full preprocessing pipeline on import.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
RAW_MIND = ROOT / "data" / "raw" / "mind"
RAW_EXTERNAL = ROOT / "data" / "raw" / "external"
PROCESSED = ROOT / "data" / "processed"

TRAIN_DIR = RAW_MIND / "train"
DEV_DIR = RAW_MIND / "dev"

ITEMS_OUT = PROCESSED / "items.csv"
USERS_OUT = PROCESSED / "users.csv"
INTERACTIONS_OUT = PROCESSED / "interactions.csv"
RISK_OUT = PROCESSED / "risk_labels.csv"
INTENT_OUT = PROCESSED / "intent_labels.csv"

NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

SENSITIVE_ITEM_KEYWORDS = [
    "mental health",
    "depression",
    "anxiety",
    "therapy",
    "counseling",
    "counselling",
    "crisis hotline",
    "suicide prevention",
    "domestic violence help",
    "sexual assault support",
    "addiction recovery",
    "self-harm recovery",
    "eating disorder recovery",
]

HARMFUL_ITEM_KEYWORDS = [
    "buy drugs",
    "make bomb",
    "suicide method",
    "kill yourself",
    "self harm tips",
    "extremist recruitment",
    "terrorist manifesto",
    "illegal firearm",
    "porn site",
    "gambling bonus",
    "meth recipe",
]

SENSITIVE_SESSION_KEYWORDS = [
    "depression",
    "anxiety",
    "therapy",
    "counseling",
    "suicide prevention",
    "crisis hotline",
    "addiction recovery",
    "mental health",
    "domestic violence",
    "sexual assault support",
]

HARMFUL_SESSION_KEYWORDS = [
    "buy drugs",
    "kill yourself",
    "suicide method",
    "make bomb",
    "extremist",
    "terrorist manifesto",
    "illegal firearm",
    "porn site",
    "meth recipe",
]


def clean_text(value) -> str:
    """Return a normalized string, matching the prototype's behavior."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value)
    return re.sub(r"\s+", " ", text).strip()


def ensure_dirs() -> None:
    """Create the processed data directory if needed."""
    PROCESSED.mkdir(parents=True, exist_ok=True)


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count substring keyword hits after normalization."""
    blob = clean_text(text).lower()
    return sum(1 for kw in keywords if kw in blob)


def parse_history(history: str) -> list[str]:
    """Parse MIND history into a list of item ids."""
    history = clean_text(history)
    if not history:
        return []
    return [x for x in history.split() if x]


def parse_impressions(impressions: str) -> list[tuple[str, int]]:
    """Parse MIND impression tokens into `(item_id, clicked)` pairs."""
    impressions = clean_text(impressions)
    if not impressions:
        return []

    pairs: list[tuple[str, int]] = []
    for token in impressions.split():
        if "-" in token:
            item_id, label = token.rsplit("-", 1)
            click = 1 if label == "1" else 0
        else:
            item_id = token
            click = 0
        pairs.append((item_id, click))
    return pairs


def stable_external_id(prefix: str, title: str, text: str) -> str:
    """Generate the same external item id scheme used by the prototype."""
    raw = f"{prefix}|{title}|{text}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def choose_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first matching column, else an empty string series."""
    for col in candidates:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def read_table_file(path: Path) -> pd.DataFrame:
    """Load a supported tabular file from the optional external pool."""
    suffix = "".join(path.suffixes).lower()

    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    if suffix.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    if suffix.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if suffix.endswith(".json"):
        return pd.read_json(path)
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def read_mind_news(split: str) -> pd.DataFrame:
    """Read `news.tsv` for a MIND split."""
    path = RAW_MIND / split / "news.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    df["split"] = split
    return df


def read_mind_behaviors(split: str) -> pd.DataFrame:
    """Read `behaviors.tsv` for a MIND split."""
    path = RAW_MIND / split / "behaviors.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    df["split"] = split
    return df


def detect_external_source(path: Path) -> str | None:
    """Infer the external source family from its filename."""
    name = path.name.lower()
    if "aegis" in name:
        return "aegis2"
    if "moral" in name or "education" in name:
        return "moral_education"
    return None


def normalize_external_items(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize optional external rows into the project item schema."""
    title = choose_series(
        df,
        ["title", "headline", "name", "question", "prompt", "instruction", "query"],
    ).map(clean_text)

    body = choose_series(
        df,
        ["text", "content", "body", "description", "response", "answer", "output", "article"],
    ).map(clean_text)

    category = choose_series(df, ["category", "topic", "label"]).map(clean_text)
    subcategory = choose_series(df, ["subcategory", "subtopic", "sub_label"]).map(clean_text)
    url = choose_series(df, ["url", "link"]).map(clean_text)

    out = pd.DataFrame(
        {
            "source": source_name,
            "category": category,
            "subcategory": subcategory,
            "title": title,
            "text": (title + " " + body).str.replace(r"\s+", " ", regex=True).str.strip(),
            "url": url,
        }
    )

    out = out[(out["title"] != "") | (out["text"] != "")].copy()

    prefix = "AEG" if source_name == "aegis2" else "MOR"
    out["item_id"] = [
        stable_external_id(prefix, t, x)
        for t, x in zip(out["title"].tolist(), out["text"].tolist())
    ]

    out = out[["item_id", "source", "category", "subcategory", "title", "text", "url"]]
    return out.drop_duplicates(subset=["item_id"]).reset_index(drop=True)


def load_optional_external_items() -> pd.DataFrame:
    """Load Aegis / moral education files if present, else return an empty frame."""
    empty = pd.DataFrame(
        columns=["item_id", "source", "category", "subcategory", "title", "text", "url"]
    )

    if not RAW_EXTERNAL.exists():
        print(f"[INFO] Optional external dir not found, skip: {RAW_EXTERNAL}")
        return empty

    candidate_files = []
    for path in RAW_EXTERNAL.rglob("*"):
        if not path.is_file():
            continue
        source_name = detect_external_source(path)
        if source_name is None:
            continue

        suffix = "".join(path.suffixes).lower()
        if suffix.endswith((".csv", ".tsv", ".json", ".jsonl", ".parquet")):
            candidate_files.append((path, source_name))

    if not candidate_files:
        print("[INFO] No Aegis/moral_education files found under data/raw/external, skip.")
        return empty

    normalized_frames = []
    for path, source_name in candidate_files:
        print(f"[INFO] Loading external file: {path}")
        try:
            raw_df = read_table_file(path)
            norm_df = normalize_external_items(raw_df, source_name)
            if not norm_df.empty:
                normalized_frames.append(norm_df)
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")

    if not normalized_frames:
        return empty

    all_external = pd.concat(normalized_frames, ignore_index=True)
    return all_external.drop_duplicates(subset=["item_id"]).reset_index(drop=True)


def build_items(train_news: pd.DataFrame, dev_news: pd.DataFrame) -> pd.DataFrame:
    """Build the `items.csv` equivalent from MIND plus optional external content."""
    news = pd.concat([train_news, dev_news], ignore_index=True)
    news = news.drop_duplicates(subset=["news_id"]).copy()

    mind_items = pd.DataFrame(
        {
            "item_id": news["news_id"].map(clean_text),
            "source": "mind",
            "category": news["category"].map(clean_text),
            "subcategory": news["subcategory"].map(clean_text),
            "title": news["title"].map(clean_text),
            "text": (
                news["title"].map(clean_text) + " " + news["abstract"].map(clean_text)
            ).str.replace(r"\s+", " ", regex=True).str.strip(),
            "url": news["url"].map(clean_text),
        }
    )

    external_items = load_optional_external_items()
    items = pd.concat([mind_items, external_items], ignore_index=True)
    return items.drop_duplicates(subset=["item_id"]).reset_index(drop=True)


def build_users(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated user stats from the combined behavior log."""
    tmp = behaviors.copy()

    tmp["user_id"] = tmp["user_id"].map(clean_text)
    tmp["session_id"] = tmp["split"].map(clean_text) + "_session_" + tmp["impression_id"].map(clean_text)
    tmp["history_clicks_total"] = tmp["history"].map(lambda x: len(parse_history(x)))
    tmp["impressions_total"] = tmp["impressions"].map(lambda x: len(parse_impressions(x)))
    tmp["clicked_impressions_total"] = tmp["impressions"].map(
        lambda x: sum(label for _, label in parse_impressions(x))
    )

    users = (
        tmp.groupby("user_id", as_index=False)
        .agg(
            num_sessions=("session_id", "count"),
            history_clicks_total=("history_clicks_total", "sum"),
            impressions_total=("impressions_total", "sum"),
            clicked_impressions_total=("clicked_impressions_total", "sum"),
            first_timestamp=("time", "min"),
            last_timestamp=("time", "max"),
        )
        .sort_values("user_id")
        .reset_index(drop=True)
    )
    return users


def build_interactions(behaviors: pd.DataFrame, item_source_map: dict[str, str]) -> pd.DataFrame:
    """Build interaction rows exactly as the current prototype does.

    Copied behavior:
    - history rows become `history_click`
    - impression rows become `impression`
    - ids are assigned sequentially as `INT_00000001`, ...
    """
    records = []

    for row in behaviors.itertuples(index=False):
        split = clean_text(row.split)
        impression_id = clean_text(row.impression_id)
        user_id = clean_text(row.user_id)
        timestamp = clean_text(row.time)
        session_id = f"{split}_session_{impression_id}"

        history_items = parse_history(row.history)
        for pos, item_id in enumerate(history_items, start=1):
            records.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "item_id": item_id,
                    "item_source": item_source_map.get(item_id, "mind"),
                    "event_type": "history_click",
                    "clicked": 1,
                    "position": pos,
                    "split": split,
                    "impression_id": impression_id,
                }
            )

        impression_items = parse_impressions(row.impressions)
        for pos, (item_id, click) in enumerate(impression_items, start=1):
            records.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "item_id": item_id,
                    "item_source": item_source_map.get(item_id, "mind"),
                    "event_type": "impression",
                    "clicked": click,
                    "position": pos,
                    "split": split,
                    "impression_id": impression_id,
                }
            )

    interactions = pd.DataFrame(records)
    interactions.insert(
        0,
        "interaction_id",
        [f"INT_{i:08d}" for i in range(1, len(interactions) + 1)],
    )
    return interactions


def infer_risk_label(
    source: str,
    category: str,
    subcategory: str,
    title: str,
    text: str,
) -> tuple[str, int, int]:
    """Infer the item risk label using the prototype's keyword heuristics."""
    source = clean_text(source).lower()
    blob = " ".join(
        [
            clean_text(category),
            clean_text(subcategory),
            clean_text(title),
            clean_text(text),
        ]
    ).lower()

    if source == "aegis2":
        return "harmful_promotional", 0, 1
    if source == "moral_education":
        return "sensitive_educational", 1, 0

    sensitive_hits = count_keyword_hits(blob, SENSITIVE_ITEM_KEYWORDS)
    harmful_hits = count_keyword_hits(blob, HARMFUL_ITEM_KEYWORDS)

    if harmful_hits > 0:
        return "harmful_promotional", sensitive_hits, harmful_hits
    if sensitive_hits > 0:
        return "sensitive_educational", sensitive_hits, harmful_hits
    return "benign", sensitive_hits, harmful_hits


def build_risk_labels(items: pd.DataFrame) -> pd.DataFrame:
    """Build `risk_labels.csv` rows from the item table."""
    rows = []
    for row in items.itertuples(index=False):
        risk_label, sensitive_hits, harmful_hits = infer_risk_label(
            row.source,
            row.category,
            row.subcategory,
            row.title,
            row.text,
        )
        rows.append(
            {
                "item_id": row.item_id,
                "source": row.source,
                "risk_label": risk_label,
                "sensitive_keyword_hits": sensitive_hits,
                "harmful_keyword_hits": harmful_hits,
            }
        )
    return pd.DataFrame(rows)


def infer_intent_label(session_text: str) -> tuple[str, int, int]:
    """Infer the session intent label using the current keyword heuristics."""
    sensitive_hits = count_keyword_hits(session_text, SENSITIVE_SESSION_KEYWORDS)
    harmful_hits = count_keyword_hits(session_text, HARMFUL_SESSION_KEYWORDS)

    if harmful_hits > 0:
        return "clearly_harmful_intent", sensitive_hits, harmful_hits
    if sensitive_hits > 0:
        return "sensitive_help_seeking", sensitive_hits, harmful_hits
    return "normal_interest", sensitive_hits, harmful_hits


def build_intent_labels(behaviors: pd.DataFrame, item_text_map: dict[str, str]) -> pd.DataFrame:
    """Build `intent_labels.csv` from session text assembled from item text."""
    rows = []

    for row in behaviors.itertuples(index=False):
        split = clean_text(row.split)
        impression_id = clean_text(row.impression_id)
        user_id = clean_text(row.user_id)
        session_id = f"{split}_session_{impression_id}"

        history_ids = parse_history(row.history)
        impression_ids = [item_id for item_id, _ in parse_impressions(row.impressions)]

        session_parts = []
        for item_id in history_ids + impression_ids:
            session_parts.append(item_text_map.get(item_id, ""))

        session_text = " ".join(x for x in session_parts if x).strip()
        intent_label, sensitive_hits, harmful_hits = infer_intent_label(session_text)

        rows.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "split": split,
                "intent_label": intent_label,
                "sensitive_keyword_hits": sensitive_hits,
                "harmful_keyword_hits": harmful_hits,
                "history_len": len(history_ids),
                "impression_len": len(impression_ids),
            }
        )

    return pd.DataFrame(rows)


def run_preprocess_pipeline() -> dict[str, pd.DataFrame]:
    """Run the original preprocessing pipeline and return all generated tables."""
    ensure_dirs()

    train_news = read_mind_news("train")
    dev_news = read_mind_news("dev")
    train_behaviors = read_mind_behaviors("train")
    dev_behaviors = read_mind_behaviors("dev")
    behaviors = pd.concat([train_behaviors, dev_behaviors], ignore_index=True)

    items = build_items(train_news, dev_news)
    users = build_users(behaviors)

    item_source_map = dict(zip(items["item_id"], items["source"]))
    item_text_map = dict(zip(items["item_id"], items["text"]))

    interactions = build_interactions(behaviors, item_source_map)
    risk_labels = build_risk_labels(items)
    intent_labels = build_intent_labels(behaviors, item_text_map)

    return {
        "items": items,
        "users": users,
        "interactions": interactions,
        "risk_labels": risk_labels,
        "intent_labels": intent_labels,
    }

