from __future__ import annotations
#人为构造更严格的安全评估场景，把 harmful 和 safe 外部候选加入 dev session。
#Manually create more stringent security assessment scenarios, and include the "harmful" and "safe" external candidates in the development session.
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
ITEMS_PATH = PROCESSED / "items.csv"
RISK_PATH = PROCESSED / "risk_labels.csv"

OUT_PATH = PROCESSED / "interactions_injected.csv"

#The original MIND dev set does not naturally expose the external harmful pool at ranking time, 
# so I explicitly inject harmful and safe candidates into every dev session. 
# This creates a more meaningful safety evaluation environment.
# 原版的 MIND 开发集在排名时并不会自然地暴露外部有害样本集，
# 所以我在每次开发测试中都会特意加入有害样本和安全样本。
# 这样就营造出一个更有意义的安全性评估环境。

#明确每个 dev session 注入 2 条 harmful + 1 条 safe，保持一定的注入密度，同时不过分稀释原始数据分布。
HARMFUL_PER_SESSION = 2
SAFE_PER_SESSION = 1


def load_data():
    interactions = pd.read_csv(
        INTERACTIONS_PATH,
        usecols=[
            "interaction_id",
            "session_id",
            "user_id",
            "timestamp",
            "item_id",
            "item_source",
            "event_type",
            "clicked",
            "position",
            "split",
            "impression_id",
        ],
    )

    items = pd.read_csv(
        ITEMS_PATH,
        usecols=["item_id", "source"],
    )

    risk = pd.read_csv(
        RISK_PATH,
        usecols=["item_id", "risk_label"],
    )

    return interactions, items, risk


def build_external_pools(items: pd.DataFrame, risk: pd.DataFrame):
    meta = items.merge(risk, on="item_id", how="left")
    meta["risk_label"] = meta["risk_label"].fillna("benign")

    harmful_pool = (
        meta[
            (meta["source"] == "aegis2")
            & (meta["risk_label"] == "harmful_promotional")
        ]["item_id"]
        .drop_duplicates()
        .tolist()
    )

    safe_pool = (
        meta[
            (meta["source"] == "moral_education")
        ]["item_id"]
        .drop_duplicates()
        .tolist()
    )

    if not harmful_pool:
        raise ValueError("No harmful pool found from aegis2.")
    if not safe_pool:
        raise ValueError("No safe pool found from moral_education.")

    source_map = dict(zip(items["item_id"], items["source"]))
    return harmful_pool, safe_pool, source_map


def next_interaction_start_id(interactions: pd.DataFrame) -> int:
    nums = (
        interactions["interaction_id"]
        .astype(str)
        .str.replace("INT_", "", regex=False)
        .astype(int)
    )
    return int(nums.max()) + 1


def build_session_base(dev_impr: pd.DataFrame) -> pd.DataFrame:
    base = (
        dev_impr.groupby(
            ["session_id", "user_id", "timestamp", "split", "impression_id"],
            as_index=False,
        )["position"]
        .max()
        .rename(columns={"position": "max_position"})
    )
    return base


def inject_rows(
    session_base: pd.DataFrame,
    source_map: dict[str, str],
    harmful_pool: list[str],
    safe_pool: list[str],
    start_id: int,
) -> pd.DataFrame:
    records = []
    harm_idx = 0
    safe_idx = 0
    cur_id = start_id

    for row in session_base.itertuples(index=False):
        pos = int(row.max_position)

        for _ in range(HARMFUL_PER_SESSION):
            item_id = harmful_pool[harm_idx % len(harmful_pool)]
            harm_idx += 1
            pos += 1

            records.append(
                {
                    "interaction_id": f"INT_{cur_id:08d}",
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "item_id": item_id,
                    "item_source": source_map.get(item_id, "aegis2"),
                    "event_type": "impression",
                    "clicked": 0,
                    "position": pos,
                    "split": row.split,
                    "impression_id": row.impression_id,
                }
            )
            cur_id += 1

        for _ in range(SAFE_PER_SESSION):
            item_id = safe_pool[safe_idx % len(safe_pool)]
            safe_idx += 1
            pos += 1

            records.append(
                {
                    "interaction_id": f"INT_{cur_id:08d}",
                    "session_id": row.session_id,
                    "user_id": row.user_id,
                    "timestamp": row.timestamp,
                    "item_id": item_id,
                    "item_source": source_map.get(item_id, "moral_education"),
                    "event_type": "impression",
                    "clicked": 0,
                    "position": pos,
                    "split": row.split,
                    "impression_id": row.impression_id,
                }
            )
            cur_id += 1

    return pd.DataFrame(records)


def main() -> None:
    print("[STEP] Loading processed files ...")
    interactions, items, risk = load_data()

    print("[STEP] Building harmful/safe external pools ...")
    harmful_pool, safe_pool, source_map = build_external_pools(items, risk)

    print(f"[INFO] harmful pool size = {len(harmful_pool):,}")
    print(f"[INFO] safe pool size    = {len(safe_pool):,}")

    dev_impr = interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()

    print("[STEP] Building dev session base ...")
    session_base = build_session_base(dev_impr)
    print(f"[INFO] dev sessions = {len(session_base):,}")

    start_id = next_interaction_start_id(interactions)

    print("[STEP] Injecting external items into dev impressions ...")
    injected = inject_rows(
        session_base=session_base,
        source_map=source_map,
        harmful_pool=harmful_pool,
        safe_pool=safe_pool,
        start_id=start_id,
    )

    print(f"[INFO] injected rows = {len(injected):,}")

    merged = pd.concat([interactions, injected], ignore_index=True)
    merged.to_csv(OUT_PATH, index=False)

    print()
    print("[DONE] Saved injected interactions to:")
    print(f"  {OUT_PATH}")
    print()
    print("[SUMMARY]")
    print(f"  original rows = {len(interactions):,}")
    print(f"  injected rows = {len(injected):,}")
    print(f"  merged rows   = {len(merged):,}")


if __name__ == "__main__":
    main()