# Startup Success Predictor

Predicts whether an angel-funded startup will go on to raise a follow-on round (Series A, B, or C+), based on historical Crunchbase data.

**Live demo:** [Streamlit App](https://startup-success-predictor-bgje29vjkdyttmwdh4fedp.streamlit.app/) — try it yourself, no setup required
**Live API docs:** [FastAPI Swagger UI](https://startup-success-predictor-68uj.onrender.com/docs)

> Note: the API runs on Render's free tier, which spins down after inactivity — the first request after idle time may take 20-30 seconds to respond.

---

## Overview

This project extends an earlier [Startup Funding Dashboard](https://github.com/sassiaziz99/startup-dashboard) (descriptive analytics) into a full predictive ML pipeline: given a startup's angel-stage funding details, it predicts the probability that the company will go on to raise a larger follow-on round.

**Pipeline:** raw Crunchbase CSVs → MySQL → cleaned & labeled dataset → trained classifier → FastAPI service → Docker container → deployed live → Streamlit frontend.

## Key Insight

`angel_amount_usd` (the size of the angel round) is by far the strongest predictor of follow-on funding success — nearly **4x more important** than any other single feature in the trained model, ahead of sector, country, and time-to-funding. This suggests early-stage funding intensity itself is a leading indicator of a startup's trajectory, more so than its industry or geography.

## Problem Definition

- **Label = 1**: the company raised an angel round *and* later raised a series-a/b/c+ round
- **Label = 0**: the company raised only an angel round, with no follow-on found in the data
- Restricted to angel rounds before **2010** to avoid censoring bias (giving companies enough time in the dataset window to plausibly raise a follow-on round)

This is a genuinely hard prediction problem — real investors use much richer signals (team background, traction, product data) that aren't available in this dataset. The results here reflect that honestly rather than overclaiming.

## Dataset

- Source: Crunchbase snapshot (public dataset, pre-2013)
- 3,840 angel-stage companies after cleaning and labeling
- Class balance: ~76% no follow-on / ~24% raised a follow-on round
- Features used: angel round amount, years between founding and angel round, sector (`category_code`), country

**Data quality note:** the raw CSV files had non-standard double-quote-wrapped rows and corrupted headers, requiring a custom parser (`src/load_to_mysql.py`) to correctly extract ~333K company records and ~53K funding rounds. Rows with missing sector/country after joining were kept and labeled `"unknown"` rather than dropped, to preserve sample size.

## Model

Two classifiers were trained and compared, using `class_weight="balanced"` to address class imbalance (since a naive "always predict no follow-on" baseline would otherwise score ~76% accuracy while being useless):

| Model | ROC-AUC |
|---|---|
| **Logistic Regression** (selected) | **0.675** |
| Random Forest | 0.628 |

Evaluated with ROC-AUC, precision/recall, and confusion matrix rather than raw accuracy, given the class imbalance.

## Tech Stack

- **Data**: MySQL (MariaDB via XAMPP), pandas
- **Modeling**: scikit-learn (Logistic Regression, Random Forest)
- **API**: FastAPI, served with Uvicorn
- **Containerization**: Docker
- **Deployment**: Render (API), Streamlit Community Cloud (frontend)
- **Frontend**: Streamlit

## Project Structure

```
startup-success-predictor/
├── src/
│   ├── load_to_mysql.py     # robust CSV loader (handles malformed source data)
│   ├── data_prep.py         # builds the labeled dataset
│   ├── train.py              # trains and evaluates models
│   ├── api.py                 # FastAPI service
│   └── export_options.py    # exports dropdown options for the frontend
├── app_frontend.py           # Streamlit frontend
├── models/model.joblib       # trained model pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running Locally

```bash
# Clone and set up
git clone https://github.com/sassiaziz99/startup-success-predictor.git  
cd startup-success-predictor
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# (Optional) rebuild the dataset and model from scratch
python src/data_prep.py
python src/train.py

# Run the API locally
uvicorn src.api:app --reload

# Run the frontend locally
streamlit run app_frontend.py
```

Or with Docker:
```bash
docker build -t startup-success-predictor .
docker run -p 8000:8000 startup-success-predictor
```

## Limitations & Honest Caveats

- Dataset is historical (pre-2013 Crunchbase snapshot) — real-world applicability to current startups is limited
- Predictive power (ROC-AUC 0.675) is modest, reflecting the genuine difficulty of predicting startup outcomes from limited early-stage signals
- `country_code` was kept unbucketed despite a long tail of rare countries; tree-based/linear models handle this reasonably but it's a known simplification
- Not intended as investment advice — this is a portfolio/learning project

## Author

Mohamed Aziz Sassi — Computer Science Engineering student at ESPRIT, Tunisia, specializing in Data Science and Machine Learning.