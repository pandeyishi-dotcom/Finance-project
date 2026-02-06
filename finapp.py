# ============================================================
# INDIAN MACRO EVENT & RISK ANALYTICS DASHBOARD
# (DEFENSIVE, INDEX-SAFE, PRODUCTION-GRADE)
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

# ---------------- REGIME ----------------
def inflation_regime(x):
    if x >= 6:
        return "High Inflation"
    elif x <= 4:
        return "Low Inflation"
    else:
        return "Mid Inflation"

# ---------------- EVENT STATE ----------------
def macro_state(date):
    release = IST.localize(datetime(date.year, date.month, date.day, 17, 30))
    now = datetime.now(IST)
    if now < release:
        return "PRE"
    elif now <= release + timedelta(minutes=60):
        return "LIVE"
    else:
        return "POST"

# ---------------- MARKET DATA ----------------
def get_market_data(ticker, event_date):
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_date = pd.to_datetime(event_date)

    try:
        if (now - event_date).days <= 30:
            df = yf.download(
                ticker,
                event_date - timedelta(hours=2),
                event_date + timedelta(hours=2),
                interval="1m",
                progress=False
            )
            if not df.empty:
                return df, "1m"

        df = yf.download(
            ticker,
            event_date - timedelta(days=2),
            event_date + timedelta(days=2),
            interval="5m",
            progress=False
        )
        if not df.empty:
            return df, "5m"

        df = yf.download(
            ticker,
            event_date - timedelta(days=10),
            event_date + timedelta(days=10),
            interval="1d",
            progress=False
        )
        return df, "1d"
    except:
        return pd.DataFrame(), "NA"

def align_event(df, event_date):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["t"] = (df.index - event_date).total_seconds() / 60
    return df.set_index("t")

def reaction_return(df):
    try:
        pre = df.loc[df.index < 0]["Close"].iloc[-1]
        post = df.loc[df.index > 0]["Close"].iloc[0]
        return float((post / pre - 1) * 100)
    except:
        return np.nan

# ---------------- DAILY RETURNS ----------------
@st.cache_data
def daily_returns(ticker):
    df = yf.download(ticker, period="5y", interval="1d", progress=False)
    r = df["Close"].pct_change()
    return r.dropna()

def var_95(x):
    return np.percentile(x, 5)

# ================= SIDEBAR =================
st.sidebar.title("📊 Macro Control Panel")

cpi = load_cpi()
cpi["Regime"] = cpi["Actual"].apply(inflation_regime)

event_date = st.sidebar.selectbox("CPI Release Date", cpi["Date"])
row = cpi[cpi["Date"] == event_date].iloc[0]

st.sidebar.metric("Inflation", row["Actual"])
st.sidebar.metric("Surprise", round(row["Surprise"], 2))
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

if macro_state(event_date) == "LIVE":
    time.sleep(30)
    st.experimental_rerun()

# ---------------- EVENT REACTION ----------------
if mode == "Event Reaction":
    sensitivities = {}

    fig, axes = plt.subplots(len(assets), 1, figsize=(10, 7), sharex=True)
    if len(assets) == 1:
        axes = [axes]

    for i, asset in enumerate(assets):
        df, freq = get_market_data(ASSETS[asset], event_date)
        if df.empty:
            sensitivities[asset] = np.nan
            axes[i].set_title(f"{asset} (no data)")
            continue

        al = align_event(df, event_date)
        axes[i].plot(al.index, al["Close"])
        axes[i].axvline(0, linestyle="--")
        axes[i].set_title(f"{asset} ({freq})")

        sensitivities[asset] = reaction_return(al)

    plt.xlabel("Minutes from Event")
    st.pyplot(fig)

    # SAFE sorting
    sens_series = pd.Series(sensitivities, dtype="float64").dropna()
    st.subheader("📊 Asset Sensitivity Ranking")
    st.dataframe(sens_series.sort_values(ascending=False))

# ---------------- SHOCK VS DRIFT ----------------
elif mode == "Shock vs Drift":
    rows = []

    for asset in assets:
        df, _ = get_market_data(ASSETS[asset], event_date)
        if df.empty:
            continue

        al = align_event(df, event_date)
        shock = al.loc[(al.index > 0) & (al.index <= 15)]["Close"].pct_change().sum()
        drift = al.loc[(al.index > 15) & (al.index <= 60)]["Close"].pct_change().sum()

        rows.append([asset, round(float(shock*100),2), round(float(drift*100),2)])

    st.table(pd.DataFrame(rows, columns=["Asset", "Shock %", "Drift %"]))

# ---------------- VOLATILITY ----------------
elif mode == "Volatility":
    df, _ = get_market_data("^INDIAVIX", event_date)
    if not df.empty:
        al = align_event(df, event_date)
        st.line_chart(al["Close"])
        vol_shock = al.loc[al.index > 0]["Close"].pct_change().sum()
        st.metric("Volatility Shock (%)", round(float(vol_shock*100),2))
    else:
        st.info("No volatility data available.")

# ---------------- MACRO VAR ----------------
elif mode == "Macro VaR":
    r = daily_returns("^NSEI")

    regime_map = cpi[["Date", "Regime"]].reset_index(drop=True)
    daily_regime = pd.Series(index=r.index, dtype="object")

    for i in range(len(regime_map)):
        start = regime_map.loc[i, "Date"]
        end = regime_map.loc[i+1, "Date"] if i+1 < len(regime_map) else r.index.max()
        daily_regime.loc[(daily_regime.index >= start) & (daily_regime.index < end)] = regime_map.loc[i, "Regime"]

    df = pd.DataFrame({
        "Returns": r,
        "Regime": daily_regime
    }).dropna()

    hi = df[df["Regime"] == "High Inflation"]["Returns"]
    lo = df[df["Regime"] == "Low Inflation"]["Returns"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("VaR 95% (High Inflation)", f"{round(var_95(hi)*100,2)}%" if len(hi)>30 else "N/A")
    with c2:
        st.metric("VaR 95% (Low Inflation)", f"{round(var_95(lo)*100,2)}%" if len(lo)>30 else "N/A")

# ---------------- STRESS TEST ----------------
else:
    shock = st.slider("Macro Shock (%)", -2.0, 2.0, 1.0)

    impact = 0.0
    for asset, w in PORTFOLIO_WEIGHTS.items():
        df, _ = get_market_data(ASSETS[asset], event_date)
        if df.empty:
            continue

        al = align_event(df, event_date)
        s = reaction_return(al)
        if not np.isnan(s):
            impact += float(w * s * shock)

    st.metric("Estimated Portfolio Impact (%)", round(impact, 2))
