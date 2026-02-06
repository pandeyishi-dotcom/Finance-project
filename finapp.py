# ============================================================
# GOAT-LEVEL MACRO EVENT & RISK ANALYTICS PLATFORM
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

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Macro Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

IST = pytz.timezone("Asia/Kolkata")

ASSETS = {
    "NIFTY 50": "^NSEI",
    "USD/INR": "USDINR=X",
    "India 10Y G-Sec": "IN10Y-GB",
    "India VIX": "^INDIAVIX"
}

PORTFOLIO = {
    "NIFTY 50": 0.5,
    "USD/INR": 0.3,
    "India 10Y G-Sec": 0.2
}

# ---------------- MACRO DATA ----------------
@st.cache_data
def load_cpi():
    df = pd.DataFrame({
        "Date": [
            "2023-08-14","2023-09-12","2023-10-12","2023-11-13","2023-12-12",
            "2024-01-12","2024-02-12","2024-03-12","2024-04-12","2024-05-13","2024-06-12"
        ],
        "Actual": [7.44,6.83,4.87,5.55,5.69,5.69,5.10,4.85,4.83,4.75,4.25],
        "Forecast": [6.80,6.70,5.10,5.60,5.40,5.80,5.20,5.00,4.90,4.80,4.30]
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df["Surprise"] = df["Actual"] - df["Forecast"]
    return df.sort_values("Date", ascending=False)

@st.cache_data
def load_us_cpi():
    df = pd.DataFrame({
        "Date": ["2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15"],
        "Surprise": [0.1,-0.1,0.2,0.0,-0.2]
    })
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# ---------------- REGIMES ----------------
def inflation_regime(x):
    if x >= 6: return "High Inflation"
    if x <= 4: return "Low Inflation"
    return "Mid Inflation"

def regime_confidence(series):
    last = series.iloc[0]
    return round(abs(last - series.mean()), 2)

# ---------------- EVENT STATE ----------------
def release_time(date, h=17, m=30):
    return IST.localize(datetime(date.year, date.month, date.day, h, m))

def macro_state(date):
    now = datetime.now(IST)
    rel = release_time(date)
    if now < rel: return "PRE"
    if rel <= now <= rel + timedelta(minutes=60): return "LIVE"
    return "POST"

# ---------------- MARKET DATA ----------------
def get_data(ticker, event_date):
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_date = pd.to_datetime(event_date)

    if (now - event_date).days <= 30:
        df = yf.download(ticker, event_date - timedelta(hours=2),
                         event_date + timedelta(hours=2),
                         interval="1m", progress=False)
        if not df.empty: return df, "1m"

    df = yf.download(ticker, event_date - timedelta(days=2),
                     event_date + timedelta(days=2),
                     interval="5m", progress=False)
    if not df.empty: return df, "5m"

    df = yf.download(ticker, event_date - timedelta(days=10),
                     event_date + timedelta(days=10),
                     interval="1d", progress=False)
    return df, "1d"

def align(df, event_date):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df["t"] = (df.index - event_date).total_seconds() / 60
    return df.set_index("t")

def shock_drift(df):
    shock = df.loc[(df.index > 0) & (df.index <= 15)]["Close"].pct_change().sum()
    drift = df.loc[(df.index > 15) & (df.index <= 60)]["Close"].pct_change().sum()
    return shock*100, drift*100

# ---------------- RISK ----------------
@st.cache_data
def daily_returns(ticker):
    d = yf.download(ticker, period="5y", interval="1d", progress=False)
    return d["Close"].pct_change().dropna()

def var95(x):
    return np.percentile(x, 5)

# ================= SIDEBAR =================
st.sidebar.title("🧠 Macro Control Panel")

cpi = load_cpi()
cpi["Regime"] = cpi["Actual"].apply(inflation_regime)

event_date = st.sidebar.selectbox("CPI Event", cpi["Date"])
state = macro_state(event_date)

row = cpi[cpi["Date"] == event_date].iloc[0]

st.sidebar.metric("CPI", row["Actual"])
st.sidebar.metric("Surprise", round(row["Surprise"],2))
st.sidebar.metric("Regime", row["Regime"])
st.sidebar.metric("State", state)

assets = st.sidebar.multiselect(
    "Assets",
    list(ASSETS.keys()),
    default=list(ASSETS.keys())
)

mode = st.sidebar.radio(
    "Mode",
    ["Event Reaction","Shock vs Drift","Volatility","Risk","Global Spillover"]
)

if state == "LIVE":
    time.sleep(30)
    st.experimental_rerun()

# ================= MAIN =================
st.title("📊 Macro Intelligence Platform")

# ---------- EVENT REACTION ----------
if mode == "Event Reaction":
    fig, ax = plt.subplots(len(assets), 1, figsize=(10,8), sharex=True)
    if len(assets) == 1: ax = [ax]

    reactions = {}

    for i,a in enumerate(assets):
        df,f = get_data(ASSETS[a], event_date)
        al = align(df, event_date)
        ax[i].plot(al.index, al["Close"])
        ax[i].axvline(0, linestyle="--")
        ax[i].set_title(f"{a} ({f})")
        reactions[a] = al

    st.pyplot(fig)

    st.subheader("📊 Asset Sensitivity Ranking")
    sens = {
        k: v.loc[v.index > 0]["Close"].pct_change().sum()*100
        for k,v in reactions.items()
    }
    st.write(pd.Series(sens).sort_values(ascending=False))

# ---------- SHOCK VS DRIFT ----------
elif mode == "Shock vs Drift":
    rows = []
    for a in assets:
        df,_ = get_data(ASSETS[a], event_date)
        al = align(df, event_date)
        s,d = shock_drift(al)
        rows.append([a,s,d])

    st.table(pd.DataFrame(rows, columns=["Asset","Shock %","Drift %"]))

# ---------- VOLATILITY ----------
elif mode == "Volatility":
    df,_ = get_data("^INDIAVIX", event_date)
    al = align(df, event_date)
    st.line_chart(al["Close"])
    st.metric("Vol Shock",
              round(al.loc[al.index>0]["Close"].pct_change().sum()*100,2))

# ---------- RISK ----------
elif mode == "Risk":
    r = daily_returns("^NSEI")
    hi = r[cpi[cpi["Regime"]=="High Inflation"]["Date"]]
    lo = r[cpi[cpi["Regime"]=="Low Inflation"]["Date"]]

    st.metric("VaR High Inflation", f"{round(var95(hi)*100,2)}%")
    st.metric("VaR Low Inflation", f"{round(var95(lo)*100,2)}%")

    shock = st.slider("Macro Shock", -2.0, 2.0, 1.0)
    impact = sum(shock * PORTFOLIO[k] for k in PORTFOLIO)
    st.metric("Portfolio Impact", round(impact,2))

# ---------- GLOBAL SPILLOVER ----------
else:
    us = load_us_cpi()
    st.write("US CPI → India Spillover")
    st.write(us)
