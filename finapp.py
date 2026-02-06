# ============================================================
# INDIAN MACRO EVENT – REAL-TIME ANALYTICS DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta
import pytz
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Macro Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TIMEZONES ----------------
IST = pytz.timezone("Asia/Kolkata")

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

# ---------------- MACRO DATA ----------------
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

# ---------------- MACRO LOGIC ----------------
def inflation_regime(cpi):
    if cpi >= 6:
        return "High Inflation"
    elif cpi <= 4:
        return "Low Inflation"
    else:
        return "Mid Regime"

def get_release_time(event_date, hour=17, minute=30):
    return IST.localize(datetime(
        event_date.year, event_date.month, event_date.day, hour, minute
    ))

def macro_state(event_date):
    now = datetime.now(IST)
    release = get_release_time(event_date)

    if now < release:
        return "PRE-EVENT"
    elif release <= now <= release + timedelta(minutes=60):
        return "LIVE"
    else:
        return "POST-EVENT"

# ---------------- MARKET DATA ----------------
def get_market_data(ticker, event_date):
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_date = pd.to_datetime(event_date)

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

    df = yf.download(
        ticker,
        start=event_date - timedelta(days=2),
        end=event_date + timedelta(days=2),
        interval="5m",
        progress=False
    )
    if not df.empty:
        return df.dropna(), "5-minute"

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

# ---------------- RISK FUNCTIONS ----------------
@st.cache_data
def daily_returns(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    return df["Close"].pct_change().dropna()

def var_95(returns):
    return np.percentile(returns, 5)

# ================= SIDEBAR =================
st.sidebar.title("📊 Macro Dashboard")

cpi = load_cpi_events()
mpc = load_rbi_mpc()
cpi["Regime"] = cpi["Actual"].apply(inflation_regime)

event_type = st.sidebar.radio("Event Type", ["CPI Release", "RBI MPC"])

if event_type == "CPI Release":
    event_date = st.sidebar.selectbox("CPI Date", cpi["Date"])
    row = cpi[cpi["Date"] == event_date].iloc[0]
    st.sidebar.metric("Inflation", row["Actual"])
    st.sidebar.metric("Surprise", round(row["Surprise"], 2))
    st.sidebar.metric("Regime", row["Regime"])
else:
    event_date = st.sidebar.selectbox("MPC Date", mpc["Date"])
    st.sidebar.metric("Policy Event", "RBI Rate Decision")

state = macro_state(event_date)
st.sidebar.subheader("⏱ Macro Status")
st.sidebar.metric("State", state)

assets_selected = st.sidebar.multiselect(
    "Assets",
    list(ASSETS.keys()),
    default=list(ASSETS.keys())
)

mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Event Reaction", "Macro VaR", "Stress Test"]
)

# ================= MAIN =================
st.title("🇮🇳 Indian Macro Analytics System")
st.caption("Event-Driven • Regime-Aware • Real-Time Capable")

# ---------------- AUTO REFRESH ----------------
if state == "LIVE":
    time.sleep(30)
    st.experimental_rerun()

# ---------------- EVENT REACTION ----------------
if mode == "Event Reaction":
    st.subheader("📈 Cross-Asset Event Reaction")

    fig, axes = plt.subplots(len(assets_selected), 1, figsize=(10, 7), sharex=True)
    if len(assets_selected) == 1:
        axes = [axes]

    sensitivities = {}

    for i, asset in enumerate(assets_selected):
        df, freq = get_market_data(ASSETS[asset], event_date)
        aligned = align_event(df, event_date)

        axes[i].plot(aligned.index, aligned["Close"])
        axes[i].axvline(0, linestyle="--")
        axes[i].set_title(f"{asset} ({freq})")

        sensitivities[asset] = reaction_return(aligned)

    plt.xlabel("Minutes from Event")
    st.pyplot(fig)

# ---------------- MACRO VAR ----------------
elif mode == "Macro VaR":
    st.subheader("⚠️ Macro-Conditioned VaR")

    returns = daily_returns("^NSEI")
    hi = returns[returns.index.isin(cpi[cpi["Regime"]=="High Inflation"]["Date"])]
    lo = returns[returns.index.isin(cpi[cpi["Regime"]=="Low Inflation"]["Date"])]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("VaR 95% (High Inflation)", f"{round(var_95(hi)*100,2)}%" if len(hi)>5 else "N/A")
    with c2:
        st.metric("VaR 95% (Low Inflation)", f"{round(var_95(lo)*100,2)}%" if len(lo)>5 else "N/A")

# ---------------- STRESS TEST ----------------
else:
    st.subheader("💥 Portfolio Stress Test")

    shock = st.slider("Macro Shock (%)", -2.0, 2.0, 1.0)

    impact = 0
    for asset, w in PORTFOLIO_WEIGHTS.items():
        df, _ = get_market_data(ASSETS[asset], event_date)
        aligned = align_event(df, event_date)
        s = reaction_return(aligned)
        if not np.isnan(s):
            impact += w * s * shock

    st.metric("Estimated Portfolio Impact (%)", round(impact, 2))
