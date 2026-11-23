"""
train.py
Trains and evaluates classifiers to predict whether an angel-funded startup
will go on to raise a follow-on round (series-a/b/c+).

Handles class imbalance via class_weight="balanced" rather than raw accuracy,
since accuracy alone would be misleading on a ~76/24 split (a model that
always predicts "no follow-on" would score ~76% accuracy while being useless).
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/labeled_dataset.csv"
MODEL_PATH = "models/model.joblib"

NUMERIC_FEATURES = ["angel_amount_usd", "years_to_angel"]
CATEGORICAL_FEATURES = ["category_code", "country_code"]
TARGET = "label"


def build_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(
        classification_report(y_test, preds, target_names=["no follow-on", "follow-on"])
    )
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }

    best_model = None
    best_score = -1
    best_name = None

    for name, clf in models.items():
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train, y_train)
        evaluate(name, pipeline, X_test, y_test)

        score = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.3f})")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    # Feature importances from the Random Forest (trained separately here,
    # regardless of which model "won", since it's the interpretable one)
    rf_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)

    feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = rf_pipeline.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\nTop 15 most important features (Random Forest):")
    print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
