"""
pipeline_prefect.py
Pipeline Prefect pour startup-success-predictor.
"""

import argparse
import subprocess
import sys

import joblib
import pandas as pd
from prefect import task, flow
from sklearn.metrics import roc_auc_score

PY_FILES = ["src/data_prep.py", "src/train.py", "pipeline_prefect.py"]
SECURITY_FILES = ["src/data_prep.py", "src/train.py", "pipeline_prefect.py"]
DATA_PATH = "data/labeled_dataset.csv"
MODEL_PATH = "models/model.joblib"


@task
def installer_dependances():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


@task
def formater_code():
    subprocess.run(["black", *PY_FILES], check=True)


@task
def qualite_code():
    subprocess.run(
        ["flake8", *PY_FILES, "--max-line-length=120", "--extend-ignore=E203,W503"],
        check=True,
    )


@task
def securite_code():
    subprocess.run(["bandit", "-r", *SECURITY_FILES, "-ll"], check=True)


@task
def tests_unitaires():
    subprocess.run(["pytest", "tests/", "-v"], check=True)


@task
def preparer_donnees():
    """Utilise le CSV existant, sinon tente MySQL."""
    import os

    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        print(f"[INFO] CSV existant trouvé ({DATA_PATH}) — pas besoin de MySQL")
        return DATA_PATH
    from src.data_prep import main as run

    run()
    return DATA_PATH


@task
def entrainer_modele():
    from src.train import main as run

    run()
    return MODEL_PATH


@task
def evaluer_modele():
    from src.train import evaluate, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
    from sklearn.model_selection import train_test_split

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    evaluate("Saved model (reloaded)", model, X_test, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return auc


@flow(name="code")
def flow_code():
    installer_dependances()
    formater_code()
    qualite_code()
    securite_code()
    tests_unitaires()


@flow(name="train")
def flow_train():
    preparer_donnees()
    entrainer_modele()


@flow(name="evaluate")
def flow_evaluate():
    evaluer_modele()


@flow(name="all")
def flow_all():
    flow_code()
    flow_train()
    flow_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow", required=True, choices=["all", "train", "evaluate", "code"]
    )
    args = parser.parse_args()
    {
        "all": flow_all,
        "train": flow_train,
        "evaluate": flow_evaluate,
        "code": flow_code,
    }[args.flow]()
