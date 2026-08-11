import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Pulsar-Net", page_icon="🌌")

data_dict = pickle.load(open("pulsar_xgb_model.pkl", "rb"))
model = data_dict["model"]
threshold = data_dict["threshold"]
feature_order = data_dict["features"]

st.title("Pulsar-Net")
st.write(
    "Pulsars are spinning neutron stars that give off radio wave beams. "
    "Radio surveys pick up thousands of candidate signals, most of which are just "
    "noise or interference. This tool uses a trained XGBoost model to guess whether "
    "a candidate signal is a **real pulsar** or **not**, based on its statistical shape."
)

st.divider()
st.subheader("Enter the signal's features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Integrated profile**")
    mean_profile = st.number_input("Mean", key="mean_profile", value=0.0)
    std_profile = st.number_input("Standard deviation", key="std_profile", value=1.0)
    kurtosis_profile = st.number_input("Excess kurtosis", key="kurtosis_profile", value=0.0)
    skewness_profile = st.number_input("Skewness", key="skewness_profile", value=0.0)

with col2:
    st.markdown("**DM-SNR curve**")
    mean_dmsnr = st.number_input("Mean", key="mean_dmsnr", value=0.0)
    std_dmsnr = st.number_input("Standard deviation", key="std_dmsnr", value=1.0)
    kurtosis_dmsnr = st.number_input("Excess kurtosis", key="kurtosis_dmsnr", value=0.0)
    skewness_dmsnr = st.number_input("Skewness", key="skewness_dmsnr", value=0.0)

st.divider()

if st.button("Predict", use_container_width=True):
    row = pd.DataFrame([{
        "mean_profile": mean_profile,
        "std_profile": std_profile,
        "kurtosis_profile": kurtosis_profile,
        "skewness_profile": skewness_profile,
        "mean_dmsnr": mean_dmsnr,
        "std_dmsnr": std_dmsnr,
        "kurtosis_dmsnr": kurtosis_dmsnr,
        "skewness_dmsnr": skewness_dmsnr,
    }])

    row["snr_robustness"] = row["mean_profile"] / (row["std_dmsnr"] + 1e-9)
    row["peak_to_noise"] = row["mean_profile"] / (row["skewness_dmsnr"] + 1e-9)
    row["sharpness_index"] = row["kurtosis_profile"] / (row["std_profile"] + 1e-9)
    row["ism_factor"] = row["mean_dmsnr"] * row["skewness_dmsnr"]

    row["pulsar_signature_score"] = row["kurtosis_profile"] * row["sharpness_index"]
    row["energy_concentration"] = row["mean_profile"] * row["sharpness_index"]
    row["log_dm_skew"] = np.sign(row["skewness_dmsnr"]) * np.log1p(row["skewness_dmsnr"].abs())

    skewed_cols = ["kurtosis_profile", "sharpness_index", "pulsar_signature_score"]
    for col in skewed_cols:
        row[f"log_{col}"] = np.sign(row[col]) * np.log1p(row[col].abs())

    X = row[feature_order]

    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= threshold)

    if pred == 1:
        st.success(f"Likely a real pulsar — {proba:.1%} confidence")
    else:
        st.error(f"Likely noise or RFI — {proba:.1%} confidence")

    st.caption(f"Model: XGBoost (threshold = {threshold:.3f}), trained on the HTRU2 dataset.")

st.divider()
st.caption("Not sure what values to try? Real pulsar candidates tend to have a low mean and skewness in the integrated profile, with a higher mean in the DM-SNR curve.")