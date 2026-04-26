"""Reusable dev-set injection helpers from `scripts/inject_external_into_dev.py`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/e/intent_aware_rec_defense")
PROCESSED = ROOT / "data" / "processed"

INTERACTIONS_PATH = PROCESSED / "interactions.csv"
ITEMS_PATH = PROCESSED / "items.csv"
RISK_PATH = PROCESSED / "risk_labels.csv"
OUT_PATH = PROCESSED / "interactions_injected.csv"

HARMFUL_PER_SESSION = 2
SAFE_PER_SESSION = 1


def load_processed_inputs(
    interactions_path: Path = INTERACTIONS_PATH,
    items_path: Path = ITEMS_PATH,
    risk_path: Path = RISK_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed inputs used by the current injection script."""
    interactions = pd.read_csv(
        interactions_path,
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

    items = pd.read_csv(items_path, usecols=["item_id", "source"])
    risk = pd.read_csv(risk_path, usecols=["item_id", "risk_label"])
    return interactions, items, risk


def build_external_pools(
    items: pd.DataFrame,
    risk: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Build harmful and safe external pools exactly as in the prototype."""
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
        meta[meta["source"] == "moral_education"]["item_id"]
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
    """Return the next numeric interaction id after the current max."""
    nums = (
        interactions["interaction_id"]
        .astype(str)
        .str.replace("INT_", "", regex=False)
        .astype(int)
    )
    return int(nums.max()) + 1


def get_dev_impressions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Filter to dev impression rows."""
    return interactions[
        (interactions["split"] == "dev")
        & (interactions["event_type"] == "impression")
    ].copy()


def build_session_base(dev_impr: pd.DataFrame) -> pd.DataFrame:
    """Build one row per dev session with the current max ranking position."""
    return (
        dev_impr.groupby(
            ["session_id", "user_id", "timestamp", "split", "impression_id"],
            as_index=False,
        )["position"]
        .max()
        .rename(columns={"position": "max_position"})
    )


def inject_rows(
    session_base: pd.DataFrame,
    source_map: dict[str, str],
    harmful_pool: list[str],
    safe_pool: list[str],
    start_id: int,
    harmful_per_session: int = HARMFUL_PER_SESSION,
    safe_per_session: int = SAFE_PER_SESSION,
) -> pd.DataFrame:
    """Append deterministic injected rows by cycling through the external pools."""
    records = []
    harm_idx = 0
    safe_idx = 0
    cur_id = start_id

    for row in session_base.itertuples(index=False):
        pos = int(row.max_position)

        for _ in range(harmful_per_session):
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

        for _ in range(safe_per_session):
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


def build_injected_interactions(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    risk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(merged, injected_only)` using the current prototype logic."""
    harmful_pool, safe_pool, source_map = build_external_pools(items, risk)
    dev_impr = get_dev_impressions(interactions)
    session_base = build_session_base(dev_impr)
    start_id = next_interaction_start_id(interactions)

    injected = inject_rows(
        session_base=session_base,
        source_map=source_map,
        harmful_pool=harmful_pool,
        safe_pool=safe_pool,
        start_id=start_id,
    )
    merged = pd.concat([interactions, injected], ignore_index=True)
    return merged, injected

