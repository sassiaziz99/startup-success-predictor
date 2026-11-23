# Résultats - Startup Success Predictor

Jeu de test : 768 startups (split 80/20 stratifié, random_state=42)

| Modèle | ROC-AUC |
|---|---|
| Logistic Regression (selected) | 0.675 |
| Random Forest | 0.628 |

## Logistic Regression (selected)
- Matrice de confusion : TN=383, FP=200, FN=75, TP=110
- Classe 1 (follow-on) : precision=0.35, recall=0.59, f1=0.44

## Random Forest
- Matrice de confusion : TN=433, FP=150, FN=105, TP=80
- Classe 1 (follow-on) : precision=0.35, recall=0.43, f1=0.39
