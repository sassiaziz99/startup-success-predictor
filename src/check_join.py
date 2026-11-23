"""
check_join.py
Diagnose why objects.id isn't matching funding_rounds.object_id.
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

objects_sample = pd.read_sql("SELECT id FROM objects LIMIT 5", engine)
rounds_sample = pd.read_sql("SELECT object_id FROM funding_rounds LIMIT 5", engine)

print("objects.id sample:")
print(objects_sample)
print("\nfunding_rounds.object_id sample:")
print(rounds_sample)

one_id = pd.read_sql("SELECT object_id FROM funding_rounds LIMIT 1", engine)["object_id"].iloc[0]
print(f"\nChecking if '{one_id}' exists in objects.id ...")
match = pd.read_sql(f"SELECT * FROM objects WHERE id = '{one_id}'", engine)
print(match)

overlap = pd.read_sql(
    """
    SELECT COUNT(*) AS matched
    FROM funding_rounds fr
    INNER JOIN objects o ON fr.object_id = o.id
    """,
    engine,
)
print("\nTotal matched rows via SQL JOIN:")
print(overlap)