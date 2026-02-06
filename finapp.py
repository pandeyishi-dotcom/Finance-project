# ============================================================
# INDIAN MACRO EVENT & RISK ANALYTICS LAB
# (FULLY FIXED – TIMEZONE + DATA SAFE)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Indian Macro Risk Lab", layout="wide")

# ---------------- CONSTANTS ----------------
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

# ---------------- EMBEDDED MACRO DATA ----------------
@st.cache_data
def load_cpi_events():
    df = pd.DataFrame({
        "Date": [
            "2023-08-14","2023-09-12","2023-10-12",
            "2023-11-13","2023-12-12",
            "2024-01-12","2024-02-12","2024-03-12",
            "2024-04-12","2024-05-13","2024-06-12"
        ],
        "Actual": [7.44,6.83,4.87,5.55,5.69,5.69,5.10,4.85,4.83,4.75,4.25],
        "Forecast": [6.80,6.70,5.10,5.60,5.40,5.80,5.20,5.00,4.90,4.80,4.30]
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df["Surprise"] = df["Actual"] - df["Forecast"]
    return df.sort_values("Date", ascending=False)

@st.cache_data
def load_rbi_mpc():
    df = pd.DataFrame({
        "Date": [
            "2023-08-10","2023-10-06","2023-12-08",
            "2024-02-08","2024-04-05","2024-06-07"
        ],
        "Policy": ["Rate Decision"] * 6
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# ---------------- REGIME LOGIC ----------------
def inflation_regime(cpi):
    if cpi >= 6:
        return "High Inflation"
    elif cpi <= 4:
        return "Low Inflation"
    else:
        return "Mid Regime"

# ---------------- ADAPTIVE MARKET DATA ----------------
def get_market_data(ticker, event_date):
    # FIX: make both timestamps timezone-naive
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_date = pd.to_datetime(event_date)

    # 1️⃣ 1-minute data (recent only)
    if (now - event_date).days <= 30:
        df = yf.download(
            ticker,
            start=event_date - timedelta(hours=2),
            end=event_date + timedelta(hours=2),
            interval="1m",
            progress=False
        )
        if not df.empty:
            return df.dropna(), "1-minute"

    # 2️⃣ 5-minute intraday (historical)
    df = yf.download(
        ticker,
        start=event_date - timedelta(days=2),
        end=event_date + timedelta(days=2),
        interval="5m",
        progress=False
    )
    if not df.empty:
        return df.dropna(), "5-minute"

    # 3️⃣ Daily fallback (always works)
    df = yf.download(
        ticker,
        start=event_date - timedelta(days=10),
        end=event_date + timedelta(days=10),
        interval="1d",
        progress=False
    )
    return df.dropna(), "daily"

def align_event(df, event_date):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["t"] = (df.index - event_date).total_seconds() / 60
    return df.set_index("t")

def reaction_return(df):
    try:
        pre = df.loc[df.index < 0]["Close"].iloc[-1]
        post = df.loc[df.index > 0]["Close"].iloc[0]
        return (post / pre - 1) * 100
    except:
        return np.nan

# ---------------- DAILY RETURNS ----------------
@st.cache_data
def daily_returns(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    return df["Close"].pct_change().dropna()

def var_95(returns):
    return np.percentile(returns, 5)

# ================= STREAMLIT UI =================
st.title("🇮🇳 Indian Macro Event & Risk Analytics Lab")
st.caption("Adaptive frequency • CPI & RBI • Regimes • Macro VaR")

cpi = load_cpi_events()
mpc = load_rbi_mpc()

cpi["Regime"] = cpi["Actual"].apply(inflation_regime)

event_type = st.selectbox("Select Event Type", ["CPI Release", "RBI MPC"])

if event_type == "CPI Release":
    event_date = st.selectbox("Select CPI Date", cpi["Date"])
    row = cpi[cpi["Date"] == event_date].iloc[0]
    st.metric("CPI Surprise", round(row["Surprise"], 2))
    st.metric("Inflation Regime", row["Regime"])
else:
    event_date = st.selectbox("Select MPC Date", mpc["Date"])
    st.metric("Policy Event", "RBI MPC Decision")

# ---------------- CROSS-ASSET REACTION ----------------
st.subheader("📊 Cross-Asset Reaction")

fig, axes = plt.subplots(len(ASSETS), 1, figsize=(10, 8), sharex=True)
sensitivities = {}

for i, (name, ticker) in enumerate(ASSETS.items()):
    data, freq = get_market_data(ticker, event_date)
    aligned = align_event(data, event_date)

    axes[i].plot(aligned.index, aligned["Close"])
    axes[i].axvline(0, linestyle="--")
    axes[i].set_title(f"{name} ({freq})")

    sensitivities[name] = reaction_return(aligned)

plt.xlabel("Minutes from Event")
st.pyplot(fig)

# ---------------- REGRESSION (DAILY – ROBUST) ----------------
st.subheader("📈 CPI Surprise → INR Reaction (Daily)")

reg_data = []
for _, row in cpi.iterrows():
    df = yf.download(
        "USDINR=X",
        start=row["Date"] - timedelta(days=1),
        end=row["Date"] + timedelta(days=2),
        interval="1d",
        progress=False
    )
    if len(df) >= 2:
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        reg_data.append([row["Surprise"], ret])

reg_df = pd.DataFrame(reg_data, columns=["Surprise", "INR_Return"])

X = sm.add_constant(reg_df["Surprise"])
model = sm.OLS(reg_df["INR_Return"], X).fit()
st.text(model.summary())

# ---------------- MACRO-CONDITIONED VAR ----------------
st.subheader("⚠️ Macro-Conditioned VaR (NIFTY)")

returns = daily_returns("^NSEI")

hi_dates = cpi[cpi["Regime"] == "High Inflation"]["Date"]
lo_dates = cpi[cpi["Regime"] == "Low Inflation"]["Date"]

hi = returns[returns.index.isin(hi_dates)]
lo = returns[returns.index.isin(lo_dates)]

c1, c2 = st.columns(2)
with c1:
    st.metric("VaR 95% (High Inflation)", f"{round(var_95(hi)*100,2)}%" if len(hi) > 5 else "N/A")
with c2:
    st.metric("VaR 95% (Low Inflation)", f"{round(var_95(lo)*100,2)}%" if len(lo) > 5 else "N/A")

# ---------------- PORTFOLIO STRESS TEST ----------------
st.subheader("💥 Portfolio Stress Test")

shock = st.slider("Macro Shock (%)", -2.0, 2.0, 1.0)

impact = 0
for asset, w in PORTFOLIO_WEIGHTS.items():
    s = sensitivities.get(asset, 0)
    if not np.isnan(s):
        impact += w * s * shock

st.metric("Estimated Portfolio Impact (%)", round(impact, 2))
