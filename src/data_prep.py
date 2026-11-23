"""
data_prep.py
Builds the labeled training dataset for the Startup Success Predictor.

Label definition (adjusted to match this dataset's actual round-type vocabulary):
  1 -> company raised an "angel" round AND later raised series-a/series-b/series-c+
  0 -> company raised only an angel round (no follow-on found in the data)

"venture" rounds are excluded from follow-on detection since the stage is ambiguous.

We restrict to angel rounds funded before CUTOFF_DATE so companies have had enough
time to raise a follow-on round before the dataset window ends (avoids censoring bias).
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

CUTOFF_DATE = "2010-12-31"  # angel rounds after this date are excluded

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)


def load_data():
    objects = pd.read_sql(
        """
        SELECT id AS object_id, name, category_code, country_code,
               founded_at, status
        FROM objects
        """,
        engine,
    )
    rounds = pd.read_sql(
        """
        SELECT object_id, funded_at, funding_round_type, raised_amount_usd
        FROM funding_rounds
        """,
        engine,
    )
    return objects, rounds


def build_labels(objects, rounds):
    rounds["funding_round_type"] = rounds["funding_round_type"].str.lower().str.strip()
    rounds["funded_at"] = pd.to_datetime(rounds["funded_at"], errors="coerce")

    # Early-stage round = angel (this dataset's equivalent of "seed")
    early_rounds = rounds[
        (rounds["funding_round_type"] == "angel")
        & (rounds["funded_at"] <= pd.Timestamp(CUTOFF_DATE))
    ].copy()

    # Keep only the earliest angel round per company
    early_rounds = early_rounds.sort_values("funded_at").drop_duplicates(
        subset="object_id", keep="first"
    )

    # Follow-on success = series-a, series-b, or series-c+ at any time
    # ("venture" excluded: ambiguous stage, not a reliable success signal)
    followon_types = {"series-a", "series-b", "series-c+"}
    followon_rounds = rounds[rounds["funding_round_type"].isin(followon_types)]
    companies_with_followon = set(followon_rounds["object_id"].unique())

    early_rounds["label"] = (
        early_rounds["object_id"].isin(companies_with_followon).astype(int)
    )

    merged = early_rounds.merge(objects, on="object_id", how="left")

    merged["founded_at"] = pd.to_datetime(merged["founded_at"], errors="coerce")
    merged["years_to_angel"] = (
        merged["funded_at"].dt.year - merged["founded_at"].dt.year
    )

    merged = merged.rename(
        columns={
            "raised_amount_usd": "angel_amount_usd",
            "funded_at": "angel_date",
        }
    )

    result = merged[
        [
            "object_id",
            "name",
            "category_code",
            "country_code",
            "founded_at",
            "angel_date",
            "angel_amount_usd",
            "years_to_angel",
            "label",
        ]
    ]

    # Fill missing categorical values with "unknown" instead of dropping rows
    result["category_code"] = result["category_code"].fillna("unknown")
    result["country_code"] = result["country_code"].fillna("unknown")

    return result


def main():
    objects, rounds = load_data()
    labeled = build_labels(objects, rounds)

    print(f"Total angel-stage companies (before {CUTOFF_DATE}): {len(labeled)}")
    print("Class balance:")
    print(labeled["label"].value_counts(normalize=True))

    os.makedirs("data", exist_ok=True)
    labeled.to_csv("data/labeled_dataset.csv", index=False)
    print("Saved to data/labeled_dataset.csv")


if __name__ == "__main__":
    main()
