from __future__ import annotations
#把 session 从硬标签升级为概率分布，表达“用户当前意图有多大概率是 normal / sensitive / harmful”。
#Upgrade the session from a hard label to a probability distribution, expressing "how likely the user's current intention is to be normal / sensitive / harmful".
import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#The intent model is session-level. 
# It uses both text features and numeric behavior features,
#  because intent is not just about what appears in text but also about how the session behaves.
#该意图模型是基于会话层面的。
# 它同时使用文本特征和数值行为特征，
# 因为意图不仅取决于文本中所呈现的内容，还取决于会话的运行方式。
ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "intent"
TABLES = ROOT / "results" / "tables"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
ITEMS_PATH = PROCESSED / "items.csv"
INTENT_LABELS_PATH = PROCESSED / "intent_labels.csv"

MODEL_PATH = MODELS / "intent_model_v1.joblib"
LABEL_MAP_PATH = MODELS / "intent_label_mapping.json"
ALL_PROBS_PATH = MODELS / "intent_probabilities_all.csv"
METRICS_PATH = TABLES / "intent_model_v1_metrics.csv"
CONFUSION_PATH = TABLES / "intent_model_v1_confusion_matrix.csv"
REPORT_PATH = TABLES / "intent_model_v1_classification_report.txt"

LABEL_ORDER = [
    "normal_interest",
    "sensitive_help_seeking",
    "clearly_harmful_intent",
]

CHUNK_SIZE = 300_000


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_dirs() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def load_items_lookup() -> dict[str, str]:
    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "title", "text", "category", "subcategory"],
    )
    for col in ["title", "text", "category", "subcategory"]:
        items[col] = items[col].map(clean_text)

    items["joined_text"] = (
        items["title"]
        + " "
        + items["text"]
        + " "
        + items["category"]
        + " "
        + items["subcategory"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return dict(zip(items["item_id"], items["joined_text"]))


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


def build_session_features_chunked(
    items_lookup: dict[str, str],
    intent_df: pd.DataFrame,
) -> pd.DataFrame:
    text_parts = defaultdict(list)
    num_history_click_rows = defaultdict(int)
    num_impression_rows = defaultdict(int)
    num_clicked_impressions = defaultdict(int)
    unique_items = defaultdict(set)

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

            if event_type == "history_click":
                num_history_click_rows[session_id] += 1
                txt = items_lookup.get(item_id, "")
                if txt:
                    text_parts[session_id].append(txt)

            elif event_type == "impression":
                num_impression_rows[session_id] += 1
                if clicked == 1:
                    num_clicked_impressions[session_id] += 1
                    txt = items_lookup.get(item_id, "")
                    if txt:
                        text_parts[session_id].append(txt)

    feat = intent_df.copy()
    feat["session_text"] = feat["session_id"].map(
        lambda s: " ".join(text_parts.get(s, []))
    )
    feat["num_history_click_rows"] = feat["session_id"].map(
        lambda s: num_history_click_rows.get(s, 0)
    )
    feat["num_impression_rows"] = feat["session_id"].map(
        lambda s: num_impression_rows.get(s, 0)
    )
    feat["num_clicked_impressions"] = feat["session_id"].map(
        lambda s: num_clicked_impressions.get(s, 0)
    )
    feat["num_unique_items"] = feat["session_id"].map(
        lambda s: len(unique_items.get(s, set()))
    )

    feat["session_text"] = feat["session_text"].fillna("")

    for col in [
        "history_len",
        "impression_len",
        "num_history_click_rows",
        "num_impression_rows",
        "num_clicked_impressions",
        "num_unique_items",
    ]:
        feat[col] = feat[col].fillna(0)

    feat["clicked_impression_rate"] = (
        feat["num_clicked_impressions"] / feat["num_impression_rows"].replace(0, 1)
    )
    feat["history_to_impression_ratio"] = (
        feat["history_len"] / feat["impression_len"].replace(0, 1)
    )

    return feat


def build_matrices(df_train: pd.DataFrame, df_dev: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_text_train = vectorizer.fit_transform(df_train["session_text"])
    X_text_dev = vectorizer.transform(df_dev["session_text"])

    num_cols = [
        "history_len",
        "impression_len",
        "num_history_click_rows",
        "num_impression_rows",
        "num_clicked_impressions",
        "num_unique_items",
        "clicked_impression_rate",
        "history_to_impression_ratio",
    ]

    X_num_train = csr_matrix(df_train[num_cols].astype(float).values)
    X_num_dev = csr_matrix(df_dev[num_cols].astype(float).values)

    X_train = hstack([X_text_train, X_num_train]).tocsr()
    X_dev = hstack([X_text_dev, X_num_dev]).tocsr()

    return vectorizer, num_cols, X_train, X_dev


def train_and_evaluate(feature_df: pd.DataFrame):
    df_train = feature_df[feature_df["split"] == "train"].copy()
    df_dev = feature_df[feature_df["split"] == "dev"].copy()

    label_to_id = {label: i for i, label in enumerate(LABEL_ORDER)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    y_train = df_train["intent_label"].map(label_to_id).values
    y_dev = df_dev["intent_label"].map(label_to_id).values

    vectorizer, num_cols, X_train, X_dev = build_matrices(df_train, df_dev)

#
#Feature Engineering Module
#特征工程模块
#概率输出：p_normal_interest,p_sensitive_help_seeking,p_clearly_harmful_intent
#The output is not just a hard label. 
# It is a probability distribution over the three intent classes, 
# and this uncertainty is later preserved in the ranking layer.
#其输出并非只是一个明确的标签。
# 它是一个针对这三种意图类别的概率分布，而这种不确定性随后会保留在排序层中。


    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    dev_pred = clf.predict(X_dev)
    dev_prob = clf.predict_proba(X_dev)

    acc = accuracy_score(y_dev, dev_pred)
    report_dict = classification_report(
        y_dev,
        dev_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_txt = classification_report(
        y_dev,
        dev_pred,
        labels=[0, 1, 2],
        target_names=LABEL_ORDER,
        zero_division=0,
    )

    cm = confusion_matrix(y_dev, dev_pred, labels=[0, 1, 2])

    metrics_rows = []
    for label in LABEL_ORDER:
        row = report_dict[label]
        metrics_rows.append(
            {
                "label": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1-score"],
                "support": row["support"],
            }
        )

    metrics_rows.append(
        {
            "label": "overall_accuracy",
            "precision": acc,
            "recall": acc,
            "f1_score": acc,
            "support": len(y_dev),
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)
    cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)

    full_df = feature_df.copy()
    X_text_full = vectorizer.transform(full_df["session_text"])
    X_num_full = csr_matrix(full_df[num_cols].astype(float).values)
    X_full = hstack([X_text_full, X_num_full]).tocsr()

    full_pred = clf.predict(X_full)
    full_prob = clf.predict_proba(X_full)

    probs_df = full_df[["session_id", "user_id", "split", "intent_label"]].copy()
    probs_df["predicted_intent_label"] = [id_to_label[i] for i in full_pred]
    probs_df["p_normal_interest"] = full_prob[:, label_to_id["normal_interest"]]
    probs_df["p_sensitive_help_seeking"] = full_prob[:, label_to_id["sensitive_help_seeking"]]
    probs_df["p_clearly_harmful_intent"] = full_prob[:, label_to_id["clearly_harmful_intent"]]

    artifact = {
        "vectorizer": vectorizer,
        "model": clf,
        "numeric_columns": num_cols,
        "label_order": LABEL_ORDER,
    }

    return artifact, metrics_df, cm_df, report_txt, probs_df


def main() -> None:
    ensure_dirs()

    print("[STEP] Loading item lookup ...")
    items_lookup = load_items_lookup()

    print("[STEP] Loading intent labels ...")
    intent_df = load_intent_labels()

    print("[STEP] Building session-level features (chunked) ...")
    feature_df = build_session_features_chunked(items_lookup, intent_df)

    print("[STEP] Training intent probability model ...")
    artifact, metrics_df, cm_df, report_txt, probs_df = train_and_evaluate(feature_df)

    print("[STEP] Saving model and evaluation outputs ...")
    joblib.dump(artifact, MODEL_PATH)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"label_order": LABEL_ORDER}, f, ensure_ascii=False, indent=2)

    metrics_df.to_csv(METRICS_PATH, index=False)
    cm_df.to_csv(CONFUSION_PATH)
    probs_df.to_csv(ALL_PROBS_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print()
    print("[DONE] Intent model outputs saved:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {LABEL_MAP_PATH}")
    print(f"  - {METRICS_PATH}")
    print(f"  - {CONFUSION_PATH}")
    print(f"  - {ALL_PROBS_PATH}")
    print(f"  - {REPORT_PATH}")
    print()
    print("[DEV METRICS]")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()