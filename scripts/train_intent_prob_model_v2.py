from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import MaxAbsScaler


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "intent"
TABLES = ROOT / "results" / "tables"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
ITEMS_PATH = PROCESSED / "items.csv"
INTENT_LABELS_PATH = PROCESSED / "intent_labels.csv"

MODEL_PATH = MODELS / "intent_model_v2.joblib"
LABEL_MAP_PATH = MODELS / "intent_label_mapping_v2.json"
ALL_PROBS_PATH = MODELS / "intent_probabilities_all_v2.csv"
METRICS_PATH = TABLES / "intent_model_v2_metrics_v2.csv"
TOP_FEATURES_PATH = TABLES / "intent_model_v2_top_features.csv"
REPORT_PATH = TABLES / "intent_model_v2_classification_report.txt"

LABEL_ORDER = [
    "normal_interest",
    "sensitive_help_seeking",
    "clearly_harmful_intent",
]

CHUNK_SIZE = 300_000

SENSITIVE_KEYWORDS = {
    "depression", "anxiety", "self-harm", "therapy", "mental health",
    "counselor", "counselling", "stress", "trauma", "suicide",
    "support", "crisis", "help", "recovery", "wellbeing",
}

HARMFUL_KEYWORDS = {
    "kill", "bomb", "attack", "hate", "weapon", "terror", "abuse",
    "violent", "violence", "self harm", "self-harm methods",
    "extremist", "radical", "harass", "harassment", "promote harm",
}


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def ensure_dirs() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def load_items_lookup():
    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "title", "text", "category", "subcategory", "source"],
    )

    for col in ["title", "text", "category", "subcategory", "source"]:
        items[col] = items[col].map(clean_text)

    items["joined_text"] = (
        items["title"] + " " +
        items["text"] + " " +
        items["category"] + " " +
        items["subcategory"] + " " +
        items["source"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    text_lookup = dict(zip(items["item_id"], items["joined_text"]))
    category_lookup = dict(zip(items["item_id"], items["category"]))
    source_lookup = dict(zip(items["item_id"], items["source"]))

    return text_lookup, category_lookup, source_lookup


def load_intent_labels() -> pd.DataFrame:
    return pd.read_csv(
        INTENT_LABELS_PATH,
        usecols=[
            "session_id",
            "user_id",
            "split",
            "intent_label",
            "history_len",
            "impression_len",
        ],
    )


def keyword_hits(text: str, vocab: set[str]) -> int:
    text = text.lower()
    return sum(1 for kw in vocab if kw in text)


def build_session_features_chunked(
    text_lookup: dict[str, str],
    category_lookup: dict[str, str],
    source_lookup: dict[str, str],
    intent_df: pd.DataFrame,
) -> pd.DataFrame:
    text_parts = defaultdict(list)
    num_history_click_rows = defaultdict(int)
    num_impression_rows = defaultdict(int)
    num_clicked_impressions = defaultdict(int)
    unique_items = defaultdict(set)
    unique_categories = defaultdict(set)
    unique_sources = defaultdict(set)
    sensitive_hits = defaultdict(int)
    harmful_hits = defaultdict(int)

    usecols = [
        "session_id",
        "item_id",
        "event_type",
        "clicked",
    ]

    reader = pd.read_csv(INTERACTIONS_PATH, usecols=usecols, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader, start=1):
        print(f"[CHUNK] processing interactions chunk {i} ...")

        chunk["session_id"] = chunk["session_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)

        for row in chunk.itertuples(index=False):
            session_id = row.session_id
            item_id = row.item_id
            event_type = row.event_type
            clicked = int(row.clicked)

            unique_items[session_id].add(item_id)

            category = category_lookup.get(item_id, "")
            source = source_lookup.get(item_id, "")
            if category:
                unique_categories[session_id].add(category)
            if source:
                unique_sources[session_id].add(source)

            use_text = False
            if event_type == "history_click":
                num_history_click_rows[session_id] += 1
                use_text = True
            elif event_type == "impression":
                num_impression_rows[session_id] += 1
                if clicked == 1:
                    num_clicked_impressions[session_id] += 1
                    use_text = True

            if use_text:
                txt = text_lookup.get(item_id, "")
                if txt:
                    text_parts[session_id].append(txt)
                    sensitive_hits[session_id] += keyword_hits(txt, SENSITIVE_KEYWORDS)
                    harmful_hits[session_id] += keyword_hits(txt, HARMFUL_KEYWORDS)

    feat = intent_df.copy()
    feat["session_text"] = feat["session_id"].map(lambda s: " ".join(text_parts.get(s, [])))
    feat["num_history_click_rows"] = feat["session_id"].map(lambda s: num_history_click_rows.get(s, 0))
    feat["num_impression_rows"] = feat["session_id"].map(lambda s: num_impression_rows.get(s, 0))
    feat["num_clicked_impressions"] = feat["session_id"].map(lambda s: num_clicked_impressions.get(s, 0))
    feat["num_unique_items"] = feat["session_id"].map(lambda s: len(unique_items.get(s, set())))
    feat["num_unique_categories"] = feat["session_id"].map(lambda s: len(unique_categories.get(s, set())))
    feat["num_unique_sources"] = feat["session_id"].map(lambda s: len(unique_sources.get(s, set())))
    feat["sensitive_keyword_hits"] = feat["session_id"].map(lambda s: sensitive_hits.get(s, 0))
    feat["harmful_keyword_hits"] = feat["session_id"].map(lambda s: harmful_hits.get(s, 0))

    feat["session_text"] = feat["session_text"].fillna("")

    for col in [
        "history_len",
        "impression_len",
        "num_history_click_rows",
        "num_impression_rows",
        "num_clicked_impressions",
        "num_unique_items",
        "num_unique_categories",
        "num_unique_sources",
        "sensitive_keyword_hits",
        "harmful_keyword_hits",
    ]:
        feat[col] = feat[col].fillna(0)

    feat["clicked_impression_rate"] = (
        feat["num_clicked_impressions"] / feat["num_impression_rows"].replace(0, 1)
    )
    feat["history_to_impression_ratio"] = (
        feat["history_len"] / feat["impression_len"].replace(0, 1)
    )
    feat["keyword_balance"] = feat["sensitive_keyword_hits"] - feat["harmful_keyword_hits"]

    return feat


def build_matrices(df_train: pd.DataFrame, df_dev: pd.DataFrame):
    word_vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        min_df=2,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=3000,
        min_df=3,
    )

    X_word_train = word_vectorizer.fit_transform(df_train["session_text"])
    X_word_dev = word_vectorizer.transform(df_dev["session_text"])

    X_char_train = char_vectorizer.fit_transform(df_train["session_text"])
    X_char_dev = char_vectorizer.transform(df_dev["session_text"])

    num_cols = [
        "history_len",
        "impression_len",
        "num_history_click_rows",
        "num_impression_rows",
        "num_clicked_impressions",
        "num_unique_items",
        "num_unique_categories",
        "num_unique_sources",
        "sensitive_keyword_hits",
        "harmful_keyword_hits",
        "clicked_impression_rate",
        "history_to_impression_ratio",
        "keyword_balance",
    ]

    scaler = MaxAbsScaler()
    X_num_train_dense = scaler.fit_transform(df_train[num_cols].astype(float).values)
    X_num_dev_dense = scaler.transform(df_dev[num_cols].astype(float).values)

    X_num_train = csr_matrix(X_num_train_dense)
    X_num_dev = csr_matrix(X_num_dev_dense)

    X_train = hstack([X_word_train, X_char_train, X_num_train]).tocsr()
    X_dev = hstack([X_word_dev, X_char_dev, X_num_dev]).tocsr()

    return word_vectorizer, char_vectorizer, scaler, num_cols, X_train, X_dev


def train_and_select(feature_df: pd.DataFrame):
    df_train = feature_df[feature_df["split"] == "train"].copy()
    df_dev = feature_df[feature_df["split"] == "dev"].copy()

    label_to_id = {label: i for i, label in enumerate(LABEL_ORDER)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    y_train = df_train["intent_label"].map(label_to_id).values
    y_dev = df_dev["intent_label"].map(label_to_id).values

    word_vectorizer, char_vectorizer, scaler, num_cols, X_train, X_dev = build_matrices(df_train, df_dev)

    best = None
    best_metrics = None
    best_report = None

    for C in [0.5, 1.0, 2.0]:
        print(f"[MODEL] training LogisticRegression with C={C} ...")
        clf = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
            C=C,
        )
        clf.fit(X_train, y_train)

        dev_pred = clf.predict(X_dev)
        macro_f1 = f1_score(y_dev, dev_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_dev, dev_pred)

        report = classification_report(
            y_dev,
            dev_pred,
            labels=[0, 1, 2],
            target_names=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )

        score = (macro_f1, acc)
        if best is None or score > best:
            best = score
            best_metrics = {
                "C": C,
                "macro_f1": macro_f1,
                "accuracy": acc,
                "clf": clf,
                "dev_pred": dev_pred,
            }
            best_report = report

    clf = best_metrics["clf"]
    dev_pred = best_metrics["dev_pred"]
    dev_prob = clf.predict_proba(X_dev)

    rows = []
    for label in LABEL_ORDER:
        row = best_report[label]
        rows.append(
            {
                "label": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": row["support"],
                "selected_C": best_metrics["C"],
                "overall_accuracy": best_metrics["accuracy"],
                "macro_f1": best_metrics["macro_f1"],
            }
        )

    rows.append(
        {
            "label": "overall_accuracy",
            "precision": best_metrics["accuracy"],
            "recall": best_metrics["accuracy"],
            "f1_score": best_metrics["accuracy"],
            "support": len(y_dev),
            "selected_C": best_metrics["C"],
            "overall_accuracy": best_metrics["accuracy"],
            "macro_f1": best_metrics["macro_f1"],
        }
    )

    metrics_df = pd.DataFrame(rows)

    full_df = feature_df.copy()

    X_word_full = word_vectorizer.transform(full_df["session_text"])
    X_char_full = char_vectorizer.transform(full_df["session_text"])
    X_num_full = scaler.transform(full_df[num_cols].astype(float).values)
    X_num_full = csr_matrix(X_num_full)

    X_full = hstack([X_word_full, X_char_full, X_num_full]).tocsr()

    full_pred = clf.predict(X_full)
    full_prob = clf.predict_proba(X_full)

    probs_df = full_df[["session_id", "user_id", "split", "intent_label"]].copy()
    probs_df["predicted_intent_label"] = [id_to_label[i] for i in full_pred]
    probs_df["p_normal_interest"] = full_prob[:, label_to_id["normal_interest"]]
    probs_df["p_sensitive_help_seeking"] = full_prob[:, label_to_id["sensitive_help_seeking"]]
    probs_df["p_clearly_harmful_intent"] = full_prob[:, label_to_id["clearly_harmful_intent"]]

    # top features
    word_names = [f"word::{x}" for x in word_vectorizer.get_feature_names_out()]
    char_names = [f"char::{x}" for x in char_vectorizer.get_feature_names_out()]
    feature_names = word_names + char_names + num_cols

    coef = clf.coef_
    top_rows = []
    for class_idx, class_name in enumerate(LABEL_ORDER):
        class_coef = coef[class_idx]
        top_idx = np.argsort(class_coef)[-20:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "class_name": class_name,
                    "rank": rank,
                    "feature_name": feature_names[idx],
                    "coefficient": class_coef[idx],
                }
            )

    top_features_df = pd.DataFrame(top_rows)

    artifact = {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "scaler": scaler,
        "model": clf,
        "numeric_columns": num_cols,
        "label_order": LABEL_ORDER,
        "selected_C": best_metrics["C"],
        "macro_f1": best_metrics["macro_f1"],
        "accuracy": best_metrics["accuracy"],
    }

    report_txt = classification_report(
        y_dev,
        dev_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        zero_division=0,
    )

    return artifact, metrics_df, probs_df, top_features_df, report_txt


def main() -> None:
    ensure_dirs()

    print("[STEP] Loading item lookups ...")
    text_lookup, category_lookup, source_lookup = load_items_lookup()

    print("[STEP] Loading intent labels ...")
    intent_df = load_intent_labels()

    print("[STEP] Building session-level features (v2) ...")
    feature_df = build_session_features_chunked(
        text_lookup=text_lookup,
        category_lookup=category_lookup,
        source_lookup=source_lookup,
        intent_df=intent_df,
    )

    print("[STEP] Training and selecting intent model v2 ...")
    artifact, metrics_df, probs_df, top_features_df, report_txt = train_and_select(feature_df)

    print("[STEP] Saving outputs ...")
    joblib.dump(artifact, MODEL_PATH)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_order": LABEL_ORDER,
                "selected_C": artifact["selected_C"],
                "macro_f1": artifact["macro_f1"],
                "accuracy": artifact["accuracy"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_df.to_csv(METRICS_PATH, index=False)
    probs_df.to_csv(ALL_PROBS_PATH, index=False)
    top_features_df.to_csv(TOP_FEATURES_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print()
    print("[DONE] Intent model v2 outputs saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {LABEL_MAP_PATH}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {ALL_PROBS_PATH}")
    print(f"  - {TOP_FEATURES_PATH}")
    print(f"  - {REPORT_PATH}")
    print()
    print("[SELECTED MODEL SUMMARY]")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
