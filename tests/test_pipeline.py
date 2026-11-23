"""Tests unitaires minimaux pour le pipeline startup-success-predictor."""
import os
import joblib
import pandas as pd

DATA_PATH = "data/labeled_dataset.csv"
MODEL_PATH = "models/model.joblib"
FEATURES = ["angel_amount_usd", "years_to_angel", "category_code", "country_code"]


def test_dataset_exists_and_not_empty():
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} introuvable"
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0
    assert "label" in df.columns


def test_label_is_binary():
    df = pd.read_csv(DATA_PATH)
    assert set(df["label"].unique()).issubset({0, 1})


def test_model_loads_and_predicts():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} introuvable"
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].head(5)
    preds = model.predict(X)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
