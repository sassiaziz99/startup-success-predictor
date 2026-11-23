"""
export_options.py
Exports just the unique category_code and country_code values from the
labeled dataset into a small JSON file, so the frontend doesn't need to
ship the full (gitignored) dataset to be deployed.
"""
import json
import pandas as pd

df = pd.read_csv("data/labeled_dataset.csv")
categories = sorted(df["category_code"].dropna().unique().tolist())
countries = sorted(df["country_code"].dropna().unique().tolist())

with open("dropdown_options.json", "w") as f:
    json.dump({"categories": categories, "countries": countries}, f, indent=2)

print(f"Saved {len(categories)} categories and {len(countries)} countries to dropdown_options.json")