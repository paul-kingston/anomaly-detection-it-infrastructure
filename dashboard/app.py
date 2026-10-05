import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="IT Infrastructure Monitoring",
    layout="wide"
)

st.title("IT Infrastructure Anomaly Monitoring")
BASE_DIR = Path(__file__).resolve().parent.parent

results = pd.read_csv(
    BASE_DIR / "results" / "lstm_anomaly_results.csv"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Samples",
        len(results)
    )

with col2:
    st.metric(
        "Detected Anomalies",
        int(results["predicted_anomaly"].sum())
    )

with col3:
    st.metric(
        "Anomaly Rate",
        f"{results['predicted_anomaly'].mean()*100:.2f}%"
    )

st.subheader("Infrastructure Anomaly Score")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    results["anomaly_score"]
)

ax.set_xlabel("Time / Sample")
ax.set_ylabel("Anomaly Score")
ax.set_title("LSTM Autoencoder Reconstruction Error")

st.pyplot(fig)

st.subheader("Detected Anomalies")

anomalies = results[
    results["predicted_anomaly"] == 1
]

st.dataframe(
    anomalies,
    use_container_width=True
)