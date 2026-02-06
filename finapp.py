# ============================================================
# INDIAN MACRO RISK & EVENT ANALYTICS LAB (SELF-CONTAINED)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Indian Macro Risk Lab", layout="wide")

# ---------------- CONSTANTS ----------------
WINDOW_MINUTES = 120

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

# ---------------- SAFE DATA LOADERS ----------------
@st.cache_data
def load_mospi_cpi():
    """
    Load CPI data.
    If CSV not found (Streamlit Cloud), use embedded MOSPI data.
    """
    if os.path.exists("india_cpi_events.csv"):
        df = pd.read_csv("india_cpi_events.csv")
    else:
        # Embedded fallback (authoritative MOSPI prints)
        df = pd.DataFrame({
            "Date": [
                "2024-01-12", "2024-02-12", "2024-03-12",
                "2024-04-12", "2024-05-13", "2024-06-12"
            ],
            "Actual": [5.69, 5.10, 4.85, 4.83, 4.75, 4.25],
            "Forecast": [5.80, 5.20, 5.00, 4.90, 4.80, 4.30]
        })

    df["Date"] = pd.to_datetime(df["Date"])
    df["Surprise"] = df["Actual"] - df["Forecast"]
    return df.sort_values("Date", ascending=False)


@st.cache_data
def load_rbi_mpc():
    """
    Load RBI MPC dates.
    Embedded so no external files are required.
    """
    df = pd.DataFrame({
        "Date": [
            "2023-02-08", "2023-04-06", "2023-06-08",
            "2023-08-10", "2023-10-06", "2023-12-08"
        ],
        "Policy": ["Rate Decision"] * 6
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------- MARKET FUNCTIONS ----------------
def get_market_data(ticker, event_time):
    start = event_time - timedelta(minutes=WINDOW_MINUTES)
    end = event_time + timedelta(minutes=WINDOW_MINUTES)
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1m",
        progress=False
    )
    return data.dropna()

def align_event(df, event_time):
    df = df.copy()
    df["t"] = (df.index - event_time).total_seconds() / 60
    return df.set_index("t")

def reaction_return(df, minutes=60):
    try:
        pre = df.loc[-30:0]["Close"].iloc[0]
        post = df.loc[0:minutes]["Close"].iloc[-1]
        return (post / pre - 1) * 100
    except:
        return np.nan

# ---------------- REGIME LOGIC ----------------
def inflation_regime(cpi):
    if cpi >= 6:
        return "High Inflation"
    elif cpi <= 4:
        return "Low Inflation"
    else:
        return "Mid Regime"

# ---------------- DAILY RETURNS ----------------
@st.cache_data
def daily_returns(ticker):
    data = yf.download(ticker, period="5y", interval="1d", progress=False)
    return data["Close"].pct_change().dropna()

def var_95(returns):
    return np.percentile(returns, 5)

# ================= STREAMLIT UI =================
st.title("🇮🇳 Indian Macro Risk & Event Analytics Lab")
st.caption("CPI • RBI MPC • Event-Time Markets • Regimes • Macro VaR")

# ---------- LOAD DATA ----------
cpi_events = load_mospi_cpi()
mpc_events = load_rbi_mpc()

cpi_events["Regime"] = cpi_events["Actual"].apply(inflation_regime)

# ---------- EVENT SELECTION ----------
event_type = st.selectbox("Select Event Type", ["CPI Release", "RBI MPC"])

if event_type == "CPI Release":
    event_date = st.selectbox("Select CPI Date", cpi_events["Date"])
    row = cpi_events[cpi_events["Date"] == event_date].iloc[0]
    st.metric("CPI Surprise", round(row["Surprise"], 2))
    st.metric("Inflation Regime", row["Regime"])
else:
    event_date = st.selectbox("Select MPC Date", mpc_events["Date"])
    st.metric("Policy Event", "RBI MPC Decision")

# ---------- CROSS-ASSET REACTION ----------
st.subheader("📊 Cross-Asset Event-Time Reaction")

fig, axes = plt.subplots(len(ASSETS), 1, figsize=(10, 8), sharex=True)
asset_sensitivity = {}

for i, (name, ticker) in enumerate(ASSETS.items()):
    data = get_market_data(ticker, event_date)

    if data.empty:
        axes[i].set_title(f"{name} (no data)")
        asset_sensitivity[name] = np.nan
        continue

    aligned = align_event(data, event_date)
    axes[i].plot(aligned.index, aligned["Close"])
    axes[i].axvline(0, linestyle="--")
    axes[i].set_title(name)

    asset_sensitivity[name] = reaction_return(aligned)

plt.xlabel("Minutes from Event")
st.pyplot(fig)

# ---------- REGRESSION ----------
st.subheader("📈 CPI Surprise → INR Reaction")

reg_data = []
for _, row in cpi_events.iterrows():
    raw = get_market_data("USDINR=X", row["Date"])
    if raw.empty:
        continue
    aligned = align_event(raw, row["Date"])
    ret = reaction_return(aligned)
    if not np.isnan(ret):
        reg_data.append([row["Surprise"], ret])

reg_df = pd.DataFrame(reg_data, columns=["Surprise", "INR_Return"])

if len(reg_df) > 5:
    X = sm.add_constant(reg_df["Surprise"])
    model = sm.OLS(reg_df["INR_Return"], X).fit()
    st.text(model.summary())
else:
    st.info("Not enough CPI events for regression.")

# ---------- MACRO-CONDITIONED VAR ----------
st.subheader("⚠️ Macro-Conditioned VaR (NIFTY)")

returns = daily_returns("^NSEI")

high_inf = cpi_events[cpi_events["Regime"] == "High Inflation"]["Date"]
low_inf = cpi_events[cpi_events["Regime"] == "Low Inflation"]["Date"]

hi = returns[returns.index.isin(high_inf)]
lo = returns[returns.index.isin(low_inf)]

col1, col2 = st.columns(2)

with col1:
    st.metric("VaR 95% (High Inflation)", f"{round(var_95(hi)*100,2)}%" if len(hi) > 10 else "N/A")

with col2:
    st.metric("VaR 95% (Low Inflation)", f"{round(var_95(lo)*100,2)}%" if len(lo) > 10 else "N/A")

# ---------- PORTFOLIO STRESS TEST ----------
st.subheader("💥 Portfolio Stress Test")

shock = st.slider("Macro Shock (%)", -2.0, 2.0, 1.0)

impact = 0
for asset, w in PORTFOLIO_WEIGHTS.items():
    s = asset_sensitivity.get(asset, 0)
    if not np.isnan(s):
        impact += w * s * shock

st.metric("Estimated Portfolio Impact (%)", round(impact, 2))
