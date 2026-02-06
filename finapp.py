# ============================================================
# CROSS-ASSET CORRELATION ANOMALY DETECTOR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cross-Asset Correlation Monitor",
    layout="wide"
)

# ---------------- ASSET UNIVERSE ----------------
ASSETS = {
    "Equities (S&P 500)": "^GSPC",
    "US Dollar (DXY)": "DX-Y.NYB",
    "US 10Y Yield": "^TNX",
    "Crude Oil (WTI)": "CL=F"
}

# ---------------- DATA LOADER ----------------
@st.cache_data
def load_prices(tickers):
    data = yf.download(
        list(tickers.values()),
        period="5y",
        interval="1d",
        progress=False
    )["Close"]
    data.columns = tickers.keys()
    return data.dropna()

# ---------------- CORRELATION LOGIC ----------------
def rolling_correlation(data, window):
    returns = data.pct_change().dropna()
    return returns.rolling(window).corr()

def correlation_zscore(rolling_corr, long_window=252):
    mean = rolling_corr.rolling(long_window).mean()
    std = rolling_corr.rolling(long_window).std()
    return (rolling_corr - mean) / std

# ================= SIDEBAR =================
st.sidebar.title("🔗 Correlation Controls")

window = st.sidebar.slider(
    "Rolling Correlation Window (days)",
    min_value=20,
    max_value=120,
    value=30
)

z_thresh = st.sidebar.slider(
    "Anomaly Threshold (|Z|)",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.1
)

# ================= MAIN =================
st.title("📊 Cross-Asset Correlation Anomaly Detector")
st.caption("Tracks when asset relationships break from historical norms")

prices = load_prices(ASSETS)
returns = prices.pct_change().dropna()

# Rolling correlations
roll_corr = rolling_correlation(prices, window)

# Z-scores of correlations
corr_z = correlation_zscore(roll_corr)

# Latest snapshot
latest_date = corr_z.index.get_level_values(0).max()
latest_corr = corr_z.loc[latest_date]

# ---------------- ANOMALY TABLE ----------------
st.subheader("🚨 Correlation Anomalies (Latest)")

anomalies = (
    latest_corr
    .reset_index()
    .rename(columns={0: "Z-Score"})
    .query("Asset1 != Asset2")
)

anomalies = anomalies[anomalies["Z-Score"].abs() >= z_thresh]
anomalies = anomalies.sort_values("Z-Score", key=abs, ascending=False)

st.dataframe(anomalies, use_container_width=True)

# ---------------- HEATMAP ----------------
st.subheader("🧠 Current Correlation Regime")

current_corr = returns.corr()

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(current_corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(current_corr.columns)))
ax.set_yticks(range(len(current_corr.columns)))
ax.set_xticklabels(current_corr.columns, rotation=45, ha="right")
ax.set_yticklabels(current_corr.columns)
fig.colorbar(im, ax=ax)
st.pyplot(fig)

# ---------------- TIME-SERIES VIEW ----------------
st.subheader("📈 Correlation Time Series")

pair = st.selectbox(
    "Select Asset Pair",
    [(a,b) for a in ASSETS for b in ASSETS if a != b]
)

pair_corr = (
    roll_corr
    .xs(pair[0], level=1)
    .xs(pair[1], level=1)
)

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(pair_corr, label="Rolling Correlation")
ax2.axhline(pair_corr.mean(), linestyle="--", color="gray", label="Long-term Mean")
ax2.set_title(f"{pair[0]} vs {pair[1]}")
ax2.legend()
st.pyplot(fig2)
