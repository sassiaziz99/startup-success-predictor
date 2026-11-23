"""
export_results.py - Évalue le modèle sauvegardé (model.joblib) et exporte les résultats.

Usage :
    python export_results.py --data labeled_dataset.csv --model model.joblib
Génère dans results/ : results.json, results.md, roc_curves.png, confusion_matrices.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from model_pipeline import build_pipeline, load_model, prepare_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="labeled_dataset.csv")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_data(a.data)

    # Modèle sauvegardé (Logistic Regression) + Random Forest pour comparaison
    lr = load_model(a.model)
    rf = Pipeline([
        ("preprocess", build_pipeline().named_steps["preprocess"]),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42)),
    ]).fit(X_train, y_train)
    models = {"Logistic Regression (selected)": lr, "Random Forest": rf}

    results, fig_roc, fig_cm = {}, plt.figure(figsize=(6, 5)), None
    fig_cm, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, m), ax in zip(models.items(), axes):
        proba = m.predict_proba(X_test)[:, 1]
        pred = m.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        results[name] = {
            "roc_auc": round(float(auc), 3),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "report": classification_report(y_test, pred, output_dict=True),
        }
        plt.figure(fig_roc.number)
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(name, fontsize=10)

    plt.figure(fig_roc.number)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)"); plt.legend(loc="lower right")
    plt.savefig(f"{a.out}/roc_curves.png", dpi=150, bbox_inches="tight")
    fig_cm.tight_layout()
    fig_cm.savefig(f"{a.out}/confusion_matrices.png", dpi=150)

    with open(f"{a.out}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# Résultats - Startup Success Predictor", "",
             f"Jeu de test : {len(y_test)} startups (split 80/20 stratifié, random_state=42)", "",
             "| Modèle | ROC-AUC |", "|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['roc_auc']} |")
    for n, r in results.items():
        c = r["confusion_matrix"]; rep = r["report"]["1"]
        lines += ["", f"## {n}",
                  f"- Matrice de confusion : TN={c[0][0]}, FP={c[0][1]}, FN={c[1][0]}, TP={c[1][1]}",
                  f"- Classe 1 (follow-on) : precision={rep['precision']:.2f}, "
                  f"recall={rep['recall']:.2f}, f1={rep['f1-score']:.2f}"]
    open(f"{a.out}/results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
