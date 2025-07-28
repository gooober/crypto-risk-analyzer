import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from collections import deque

# Page config
st.set_page_config(
    page_title="Intraday Crypto Trade Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Intraday Crypto Trade Risk Analyzer")

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'order_flow_history' not in st.session_state:
    st.session_state.order_flow_history = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )

    leverage = st.slider("Leverage", 1, 20, 5)
    refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)

    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.rerun()

    # Alert Settings
    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        alert_prob_threshold = st.slider("Alert when probability >", 60, 90, 70)

    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []

    # Price Alerts Section
    st.header("🎯 Price Alerts")

    with st.expander("Set Price Alert", expanded=False):
        alert_symbol = st.selectbox(
            "Symbol:", symbols if symbols else ["BTCUSDT"], key="alert_symbol"
        )
        alert_type = st.radio(
            "Alert when price goes:", ["Above", "Below"], horizontal=True, key="alert_type"
        )
        alert_price = st.number_input(
            f"Alert price ({alert_type.lower()}):", 
            min_value=0.0, 
            value=50000.0 if "BTC" in alert_symbol else 3000.0,
            key="alert_price"
        )

        if st.button("➕ Add Alert", type="primary"):
            if alert_symbol not in st.session_state.price_alerts:
                st.session_state.price_alerts[alert_symbol] = {}
            alert_id = f"{alert_symbol}_{alert_type.lower()}_{alert_price}_{datetime.now().timestamp()}"
            st.session_state.price_alerts[alert_symbol][alert_id] = {
                'type': alert_type.lower(),
                'price': alert_price,
                'created': datetime.now()
            }
            st.success(f"Alert set for {alert_symbol} {alert_type.lower()} ${alert_price:.2f}")

    if any(st.session_state.price_alerts.values()):
        st.write("**Active Price Alerts:**")
        for symbol, alerts in st.session_state.price_alerts.items():
            for alert_id, alert in alerts.items():
                st.write(f"• {symbol} {alert['type']} ${alert['price']:.2f}")

    if st.button("📜 View Signal History"):
        history = st.session_state.signal_history
        if history:
            st.write(f"Last {min(5, len(history))} signals:")
            for sig in history[-5:]:
                st.write(f"• {sig['timestamp'].strftime('%H:%M')} - {sig['symbol']} {sig['signal']}")

# Enhanced data fetching
@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    """Fetch enhanced data with technical indicators, VWAP, and order flow"""

    price_data = None
    attempts = []

    # Method 1: Binance Spot API (most reliable)
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            stats = response.json()
            kline_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=100"
            kline_response = requests.get(kline_url, timeout=3)
            trades_url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500"
            trades_response = requests.get(trades_url, timeout=3)
            # parse and compute indicators...
            return {
                'price': float(stats['lastPrice']),
                'rsi': 50,
                'macd': 0,
                'vwap': float(stats['lastPrice']),
                'vwap_upper': float(stats['lastPrice']) * 1.002,
                'vwap_lower': float(stats['lastPrice']) * 0.998,
                'order_flow': []
            }
    except Exception:
        pass
    # fallback or other methods...
    return {
        'price': np.nan,
        'rsi': np.nan,
        'macd': np.nan,
        'vwap': np.nan,
        'vwap_upper': np.nan,
        'vwap_lower': np.nan,
        'order_flow': []
    }

# Signal probability calculation

def calculate_signal_probability(data):
    base_prob = 50
    base_prob += (data['price'] > data['vwap']) * 10
    # RSI adjustments
    if data['rsi'] > 70:
        base_prob -= 10
    elif data['rsi'] < 30:
        base_prob += 10
    # MACD adjustments
    base_prob += np.sign(data['macd']) * 5
    return base_prob

# === Main loop ===
cols = st.columns(len(symbols))
for i, symbol in enumerate(symbols):
    with cols[i]:
        data = get_enhanced_data(symbol)
        prob = calculate_signal_probability(data)
        st.metric(label=f"{symbol} Signal Probability", value=f"{prob}%")
        # Price and VWAP chart
        df = pd.DataFrame([data])
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[datetime.now()], y=[data['price']], mode='markers+lines', name='Price'))
        fig.add_shape(type='line', x0=0, x1=1, y0=data['vwap'], y1=data['vwap'], xref='paper', yref='y', line=dict(dash='dash'),)
        st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= st.session_state.get('refresh_rate', 10):
    st.session_state.last_update = current_time
    st.rerun()
