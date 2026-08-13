"""
app_frontend.py
Simple Streamlit form that calls the live Startup Success Predictor API
and displays the prediction.
"""

import requests
import streamlit as st
import json

# Live deployed API on Render
API_URL = "https://startup-success-predictor-68uj.onrender.com/predict"

st.set_page_config(page_title="Startup Success Predictor", page_icon="$")

st.title("Startup Success Predictor")
st.write(
    "Predicts whether an angel-funded startup will go on to raise a "
    "follow-on round (Series A, B, or C+)."
)


@st.cache_data
def load_options():
    """Load sector/country options from the small pre-exported JSON file
    (see src/export_options.py) so we don't need to ship the full dataset."""
    with open("dropdown_options.json") as f:
        data = json.load(f)
    return data["categories"], data["countries"]


category_options, country_options = load_options()

with st.form("prediction_form"):
    angel_amount_usd = st.number_input(
        "Angel round amount (USD)", min_value=0, value=500000, step=10000
    )
    years_to_angel = st.number_input(
        "Years between founding and angel round", min_value=0.0, value=1.0, step=0.2
    )
    category_code = st.selectbox("Sector / category", options=category_options)
    country_code = st.selectbox("Country code", options=country_options)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "angel_amount_usd": angel_amount_usd,
        "years_to_angel": years_to_angel,
        "category_code": category_code,
        "country_code": country_code,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()

        probability = result["probability"]
        will_raise = result["will_raise_followon"]

        st.subheader("Result")
        if will_raise:
            st.success(f"✅ Likely to raise a follow-on round — {probability:.1%} probability")
        else:
            st.warning(f"⚠️ Unlikely to raise a follow-on round — {probability:.1%} probability")

        st.progress(probability)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the prediction API: {e}")

st.caption(
    "Note: this model is based on historical Crunchbase data (pre-2011) and angel round outcomes. "
    "Predictions reflect patterns in that dataset, not investment advice."
)