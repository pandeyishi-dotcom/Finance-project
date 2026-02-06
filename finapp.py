# ============================================================
# INDIAN MACRO EVENT IMPACT LAB (FIXED & ROBUST)
# ============================================================

import streamlit as st
import tradingeconomics as te
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Macro Impact Tracker",
    layout="wide"
)

# ---------------- API LOGIN ----------------
# Strongly recommended: replace with your own key
te.login("guest:guest")

# ---------------- CONFIG ----------------
ASSETS = {
    "USD/INR": "USDINR=X",
    "NIFTY 50": "^NSEI",
    "India 10Y G-Sec": "IN10Y-GB"
}

PORTFOLIO_WEIGHTS = {
    "USD/INR": 0.25,
    "NIFTY 50": 0.50,
    "India 10Y G-Sec": 0.25
}

WINDOW_MINUTES = 120

# ---------------- FUNCTIONS ----------------
@st.cache_data
def get_cpi_events():
    """
    Pull Indian CPI events safely from TradingEconomics.
    Falls back gracefully if API returns bad or empty data.
    """
    try:
        data = te.getCalendarData(
            country="India",
            initDate="2023-01-01"
        )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Defensive checks
        required_cols = {"Date", "Actual", "Forecast", "Event"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame()

        # Filter CPI events manually (API naming is inconsistent)
        df = df[df["Event"].str.contains("CPI", case=False, na=False)]

        df = df[["Date", "Actual", "Forecast"]].dropna()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Surprise"] = df["Actual"] - df["Forecast"]

        return df.sort_values("Date", ascending=False)

    except Exception:
        return pd.DataFrame()

def get_market_data(ticker, event_time):
    start = event_time - timedelta(minutes=WINDOW_MINUTES)
    end = event_time + timedelta(minutes=WINDOW_MINUTES)

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1m",
            progress=False
        )
        return data.dropna()
    except Exception:
        return pd.DataFrame()

def align_event_time(df, event_time):
    df = df.copy()
    df["t"] = (df.index - event_time).total_seconds() / 60
    return df.set_index("t")

def reaction_return(df, minutes=60):
    try:
        pre = df.loc[-30:0]["Close"].iloc[0]
        post = df.loc[0:minutes]["Close"].iloc[-1]
        return (post / pre - 1) * 100
    except Exception:
        return np.nan

# ---------------- UI ----------------
st.title("🇮🇳 Indian Macro Event Impact Tracker")
st.caption("CPI → INR → Equities → G-Secs | Event-time analysis")

events = get_cpi_events()

if events.empty:
    st.warning("No CPI data available from TradingEconomics API.")
    st.stop()

event_date = st.selectbox("Select CPI Release Date", events["Date"])
event_row = events[events["Date"] == event_date].iloc[0]

st.metric("CPI Surprise", round(event_row["Surprise"], 2))

# ---------------- CROSS-ASSET REACTION ----------------
st.subheader("📊 Cross-Asset Reaction (Event-Time Aligned)")

fig, axes = plt.subplots(len(ASSETS), 1, figsize=(10, 8), sharex=True)
asset_returns = {}

for i, (name, ticker) in enumerate(ASSETS.items()):
    raw = get_market_data(ticker, event_date)

    if raw.empty:
        axes[i].set_title(f"{name} (no data)")
        asset_returns[name] = np.nan
        continue

    aligned = align_event_time(raw, event_date)

    axes[i].plot(aligned.index, aligned["Close"])
    axes[i].axvline(0, linestyle="--")
    axes[i].set_title(name)

    asset_returns[name] = reaction_return(aligned)

plt.xlabel("Minutes from CPI Release")
st.pyplot(fig)

# ---------------- REGRESSION ----------------
st.subheader("📈 CPI Surprise → INR Reaction Regression")

regression_data = []

for _, row in events.iterrows():
    raw = get_market_data("USDINR=X", row["Date"])
    if raw.empty:
        continue

    aligned = align_event_time(raw, row["Date"])
    ret = reaction_return(aligned)

    if not np.isnan(ret):
        regression_data.append([row["Surprise"], ret])

reg_df = pd.DataFrame(regression_data, columns=["Surprise", "INR_Return"])

if len(reg_df) < 5:
    st.info("Not enough observations for regression.")
else:
    X = sm.add_constant(reg_df["Surprise"])
    model = sm.OLS(reg_df["INR_Return"], X).fit()
    st.text(model.summary())

# ---------------- STRESS TEST ----------------
st.subheader("⚠️ Portfolio Stress Test (Macro Shock)")

shock = st.slider("Assumed CPI Shock (%)", -2.0, 2.0, 1.0)

portfolio_impact = 0
for asset, weight in PORTFOLIO_WEIGHTS.items():
    sensitivity = asset_returns.get(asset, 0)
    if not np.isnan(sensitivity):
        portfolio_impact += weight * sensitivity * shock

st.metric("Estimated Portfolio Impact (%)", round(portfolio_impact, 2))

st.caption("Stress test uses historical CPI reaction sensitivities")
