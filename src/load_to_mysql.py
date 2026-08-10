"""
load_to_mysql.py
Loads objects.csv and funding_rounds.csv into MySQL.
"""

import csv
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

OBJECTS_CSV = r"C:\Users\LENOVO\Desktop\startup-dashboard-data\objects.csv"
FUNDING_ROUNDS_CSV = r"C:\Users\LENOVO\Desktop\startup-dashboard-data\funding_rounds.csv"


def unwrap_line(line: str) -> str:
    line = line.rstrip("\n").rstrip("\r")
    line = line.rstrip(";")
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
        line = line.replace('""', '"')
    return line


def load_csv_fixed(path, expected_ncols, col_names):
    rows = []
    skipped = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        reader = csv.reader((unwrap_line(line) for line in f))
        for fields in reader:
            if len(fields) == expected_ncols:
                rows.append(fields)
            else:
                skipped += 1
    print(f"  Parsed {len(rows)} good rows, skipped {skipped} malformed rows")
    return pd.DataFrame(rows, columns=col_names)


OBJECTS_ALL_NAMES = [
    "id", "entity_type", "entity_id", "parent_id", "name", "normalized_name",
    "permalink", "category_code", "status", "founded_at", "closed_at", "domain",
    "homepage_url", "twitter_username", "logo_url", "logo_width", "logo_height",
    "short_description", "description", "overview", "tag_list", "country_code",
    "state_code", "city", "region", "first_investment_at", "last_investment_at",
    "investment_rounds", "invested_companies", "first_funding_at", "last_funding_at",
    "funding_rounds", "funding_total_usd", "first_milestone_at", "last_milestone_at",
    "milestones", "relationships", "created_by", "created_at", "updated_at"
]
OBJECTS_KEEP = ["id", "name", "category_code", "status", "founded_at", "country_code"]

FUNDING_ROUNDS_ALL_NAMES = [
    "id", "funding_round_id", "object_id", "funded_at", "funding_round_type",
    "funding_round_code", "raised_amount_usd", "raised_amount", "raised_currency_code",
    "pre_money_valuation_usd", "pre_money_valuation", "pre_money_currency_code",
    "post_money_valuation_usd", "post_money_valuation", "post_money_currency_code",
    "participants", "is_first_round", "is_last_round", "source_url",
    "source_description", "created_by", "created_at", "updated_at"
]
FUNDING_ROUNDS_KEEP = ["object_id", "funded_at", "funding_round_type", "raised_amount_usd"]


def load_objects():
    print("Reading objects.csv (fixed parser) ...")
    df = load_csv_fixed(OBJECTS_CSV, len(OBJECTS_ALL_NAMES), OBJECTS_ALL_NAMES)
    df = df[OBJECTS_KEEP]
    print(f"  Final: {len(df)} rows, columns: {list(df.columns)}")
    print(df.head(3))

    df.to_sql("objects", engine, if_exists="replace", index=False, chunksize=5000)
    print("  Loaded into MySQL table 'objects'")


def load_funding_rounds():
    print("Reading funding_rounds.csv (fixed parser) ...")
    df = load_csv_fixed(FUNDING_ROUNDS_CSV, len(FUNDING_ROUNDS_ALL_NAMES), FUNDING_ROUNDS_ALL_NAMES)
    df = df[FUNDING_ROUNDS_KEEP]
    print(f"  Final: {len(df)} rows, columns: {list(df.columns)}")
    print(df.head(3))

    df.to_sql("funding_rounds", engine, if_exists="replace", index=False, chunksize=5000)
    print("  Loaded into MySQL table 'funding_rounds'")


if __name__ == "__main__":
    load_objects()
    load_funding_rounds()
    print("Done.")