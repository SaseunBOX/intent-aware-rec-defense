from __future__ import annotations
#把 item 从硬标签升级为概率分布，表达内容的风险强弱，而不是只做简单分类。
#Upgrade the "item" from a hard label to a probability distribution, expressing the degree of risk and strength of the content, rather than simply making a simple classification.
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#The risk model is item-level. 
# It concatenates title, body text, category, subcategory, 
# and source tag, then applies TF-IDF plus class-balanced logistic regression.
#该风险模型是按项目级别构建的。
# 它将标题、正文内容、类别、子类别以及来源标签进行组合，
# 然后运用 TF-IDF 方法加上具有类别平衡性的逻辑回归算法。


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models" / "risk"
TABLES = ROOT / "results" / "tables"

ITEMS_PATH = PROCESSED / "items.csv"
RISK_LABELS_PATH = PROCESSED / "risk_labels.csv"

MODEL_PATH = MODELS / "risk_model_v1.joblib"
LABEL_MAP_PATH = MODELS / "risk_label_mapping.json"
ALL_PROBS_PATH = MODELS / "risk_probabilities_all.csv"
METRICS_PATH = TABLES / "risk_model_v1_metrics.csv"
CONFUSION_PATH = TABLES / "risk_model_v1_confusion_matrix.csv"
REPORT_PATH = TABLES / "risk_model_v1_classification_report.txt"

LABEL_ORDER = [
    "benign",
    "sensitive_educational",
    "harmful_promotional",
]


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def ensure_dirs() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

#  The item-risk table is the core training data for the risk model. 
#  It combines item metadata with risk labels, 
# and also constructs a "model_text" field that concatenates all relevant text for modeling.
# 这个 item-risk 表是风险模型的核心训练数据。
# 它结合了商品元数据和风险标签，
#并且还构建了一个“模型文本”字段，该字段会将所有用于建模的相关文本进行拼接。
def load_item_risk_table() -> pd.DataFrame:
    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "source", "category", "subcategory", "title", "text"],
    )
    risk = pd.read_csv(
        RISK_LABELS_PATH,
        usecols=["item_id", "risk_label"],
    )

    df = items.merge(risk, on="item_id", how="inner")

    for col in ["source", "category", "subcategory", "title", "text", "risk_label"]:
        df[col] = df[col].map(clean_text)
#  The "model_text" field is a simple concatenation of all text-based features.
#  This allows the TF-IDF vectorizer to capture signals 
# from any of these fields without needing separate feature engineering for each. 
# “model_text”字段是所有基于文本的特征的简单拼接。
# 这使得 TF-IDF 向量化器能够从这些字段中的任何一个中捕捉到信息，而无需为每个字段进行单独的特征工程。
    df["model_text"] = (
        df["title"]
        + " "
        + df["text"]
        + " "
        + df["category"]
        + " "
        + df["subcategory"]
        + " "
        + df["source"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def train_and_evaluate(df: pd.DataFrame):
    label_to_id = {label: i for i, label in enumerate(LABEL_ORDER)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    train_df, dev_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["risk_label"],
    )

    y_train = train_df["risk_label"].map(label_to_id).values
    y_dev = dev_df["risk_label"].map(label_to_id).values
#  The TF-IDF vectorizer is configured to capture unigrams and bigrams,
# with a maximum feature limit to control dimensionality.
# TF-IDF 向量化器被设置为能够捕捉单词和双词组合，
# 并设有最大特征数量限制以控制维度。
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_df["model_text"])
    X_dev = vectorizer.transform(dev_df["model_text"])
#  The logistic regression model is trained 
# with class balancing to handle any potential label imbalance in the data.
# 逻辑回归模型使用类别平衡进行训练，以处理数据中可能存在的标签不平衡问题。
    clf = LogisticRegression(
        max_iter=2000,
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

    # 导出所有 item 的概率
    #q_benign,q_sensitive_educational,q_harmful_promotional
    X_all = vectorizer.transform(df["model_text"])
    all_pred = clf.predict(X_all)
    all_prob = clf.predict_proba(X_all)

    probs_df = df[["item_id", "source", "risk_label"]].copy()
    probs_df["predicted_risk_label"] = [id_to_label[i] for i in all_pred]
    probs_df["q_benign"] = all_prob[:, label_to_id["benign"]]
    probs_df["q_sensitive_educational"] = all_prob[:, label_to_id["sensitive_educational"]]
    probs_df["q_harmful_promotional"] = all_prob[:, label_to_id["harmful_promotional"]]

    artifact = {
        "vectorizer": vectorizer,
        "model": clf,
        "label_order": LABEL_ORDER,
    }

    return artifact, metrics_df, cm_df, report_txt, probs_df


def main() -> None:
    ensure_dirs()

    print("[STEP] Loading item-risk table ...")
    df = load_item_risk_table()

    print("[STEP] Training risk probability model ...")
    artifact, metrics_df, cm_df, report_txt, probs_df = train_and_evaluate(df)

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
    print("[DONE] Risk model outputs saved:")
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