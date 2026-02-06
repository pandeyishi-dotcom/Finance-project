# ============================================================
# INDIAN MACRO EVENT & RISK ANALYTICS DASHBOARD
# (CLEAN, REGIME-CORRECT, PRODUCTION-SAFE)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Indian Macro Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- TIMEZONE ----------------
IST = pytz.timezone("Asia/Kolkata")

# ---------------- ASSETS ----------------
ASSETS = {
    "NIFTY 50": "^NSEI",
    "USD/INR": "USDINR=X",
    "India 10Y G-Sec": "IN10Y-GB",
    "India VIX": "^INDIAVIX"
}

PORTFOLIO_WEIGHTS = {
    "NIFTY 50": 0.5,
    "USD/INR": 0.3,
    "India 10Y G-Sec": 0.2
}

# ---------------- CPI DATA ----------------
@st.cache_data
def load_cpi():
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
    return df.sort_values("Date")

# ---------------- REGIME LOGIC ----------------
def inflation_regime(value):
    if value >= 6:
        return "High Inflation"
    elif value <= 4:
        return "Low Inflation"
    else:
        return "Mid Inflation"

# ---------------- EVENT STATE ----------------
def release_time(date, hour=17, minute=30):
    return IST.localize(datetime(date.year, date.month, date.day, hour, minute))

def macro_state(date):
    now = datetime.now(IST)
    rel = release_time(date)
    if now < rel:
        return "PRE-EVENT"
    elif rel <= now <= rel + timedelta(minutes=60):
        return "LIVE"
    else:
        return "POST-EVENT"

# ---------------- MARKET DATA ----------------
def get_market_data(ticker, event_date):
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_date = pd.to_datetime(event_date)

    # Recent → minute data
    if (now - event_date).days <= 30:
        df = yf.download(
            ticker,
            start=event_date - timedelta(hours=2),
            end=event_date + timedelta(hours=2),
            interval="1m",
            progress=False
        )
        if not df.empty:
            return df, "1m"

    # Historical intraday
    df = yf.download(
        ticker,
        start=event_date - timedelta(days=2),
        end=event_date + timedelta(days=2),
        interval="5m",
        progress=False
    )
    if not df.empty:
        return df, "5m"

    # Daily fallback
    df = yf.download(
        ticker,
        start=event_date - timedelta(days=10),
        end=event_date + timedelta(days=10),
        interval="1d",
        progress=False
    )
    return df, "1d"

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

# ================= SIDEBAR =================
st.sidebar.title("📊 Macro Control Panel")

cpi = load_cpi()
cpi["Regime"] = cpi["Actual"].apply(inflation_regime)

event_date = st.sidebar.selectbox("CPI Release Date", cpi["Date"])
row = cpi[cpi["Date"] == event_date].iloc[0]

st.sidebar.metric("Inflation (%)", row["Actual"])
st.sidebar.metric("CPI Surprise", round(row["Surprise"], 2))
st.sidebar.metric("Regime", row["Regime"])
st.sidebar.metric("State", macro_state(event_date))

assets = st.sidebar.multiselect(
    "Assets",
    list(ASSETS.keys()),
    default=list(ASSETS.keys())
)

mode = st.sidebar.radio(
    "Mode",
    ["Event Reaction", "Shock vs Drift", "Volatility", "Macro VaR", "Stress Test"]
)

# ================= MAIN =================
st.title("🇮🇳 Indian Macro Analytics Dashboard")
st.caption("Event-driven • Regime-aware • Risk-focused")

# ---------------- AUTO REFRESH ----------------
if macro_state(event_date) == "LIVE":
    time.sleep(30)
    st.experimental_rerun()

# ---------------- EVENT REACTION ----------------
if mode == "Event Reaction":
    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 7), sharex=True)
    if len(assets) == 1:
        axes = [axes]

    sensitivities = {}

    for i, asset in enumerate(assets):
        df, freq = get_market_data(ASSETS[asset], event_date)
        aligned = align_event(df, event_date)

        axes[i].plot(aligned.index, aligned["Close"])
        axes[i].axvline(0, linestyle="--")
        axes[i].set_title(f"{asset} ({freq})")

        sensitivities[asset] = reaction_return(aligned)

    plt.xlabel("Minutes from Event")
    st.pyplot(fig)

    st.subheader("📊 Asset Sensitivity Ranking")
    st.write(pd.Series(sensitivities).sort_values(ascending=False))

# ---------------- SHOCK VS DRIFT ----------------
elif mode == "Shock vs Drift":
    rows = []

    for asset in assets:
        df, _ = get_market_data(ASSETS[asset], event_date)
        al = align_event(df, event_date)

        shock = al.loc[(al.index > 0) & (al.index <= 15)]["Close"].pct_change().sum() * 100
        drift = al.loc[(al.index > 15) & (al.index <= 60)]["Close"].pct_change().sum() * 100

        rows.append([asset, round(shock,2), round(drift,2)])

    st.table(pd.DataFrame(rows, columns=["Asset", "Shock %", "Drift %"]))

# ---------------- VOLATILITY ----------------
elif mode == "Volatility":
    df, _ = get_market_data("^INDIAVIX", event_date)
    al = align_event(df, event_date)

    st.line_chart(al["Close"])
    vol_shock = al.loc[al.index > 0]["Close"].pct_change().sum() * 100
    st.metric("Volatility Shock (%)", round(vol_shock, 2))

# ---------------- MACRO VAR (FIXED) ----------------
elif mode == "Macro VaR":
    returns = daily_returns("^NSEI")

    regime_map = cpi[["Date", "Regime"]].copy()
    regime_map = regime_map.sort_values("Date")

    daily_regime = pd.Series(index=returns.index, dtype="object")

    for i in range(len(regime_map)):
        start = regime_map.iloc[i]["Date"]
        end = (
            regime_map.iloc[i + 1]["Date"]
            if i + 1 < len(regime_map)
            else returns.index.max()
        )
        daily_regime.loc[(daily_regime.index >= start) & (daily_regime.index < end)] = regime_map.iloc[i]["Regime"]

    data = pd.DataFrame({
        "Returns": returns,
        "Regime": daily_regime
    }).dropna()

    hi = data[data["Regime"] == "High Inflation"]["Returns"]
    lo = data[data["Regime"] == "Low Inflation"]["Returns"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("VaR 95% (High Inflation)", f"{round(var_95(hi)*100,2)}%" if len(hi) > 20 else "N/A")
    with c2:
        st.metric("VaR 95% (Low Inflation)", f"{round(var_95(lo)*100,2)}%" if len(lo) > 20 else "N/A")

# ---------------- STRESS TEST ----------------
else:
    shock = st.slider("Macro Shock (%)", -2.0, 2.0, 1.0)

    impact = 0
    for asset, w in PORTFOLIO_WEIGHTS.items():
        df, _ = get_market_data(ASSETS[asset], event_date)
        al = align_event(df, event_date)
        s = reaction_return(al)
        if not np.isnan(s):
            impact += w * s * shock

    st.metric("Estimated Portfolio Impact (%)", round(impact, 2))
