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

# === Indicator helper functions ===

def calculate_vwap(klines):
    """Calculate VWAP and upper/lower bands from klines list"""
    volumes = [float(k[5]) for k in klines]
    typicals = [ (float(k[2]) + float(k[3]) + float(k[4]))/3 for k in klines ]
    tp_vol = [t*v for t,v in zip(typicals, volumes)]
    cum_tp_vol = np.sum(tp_vol)
    cum_vol = np.sum(volumes)
    vwap = cum_tp_vol / cum_vol if cum_vol else np.nan
    return vwap, vwap * 1.002, vwap * 0.998


def calculate_order_flow_imbalance(trades):
    """Compute order flow imbalance from trades list"""
    buy_vol = 0.0
    sell_vol = 0.0
    for trade in trades:
        qty = float(trade.get('qty', trade.get('quantity', 0)))
        # isBuyerMaker True indicates seller initiated
        if trade.get('isBuyerMaker', False):
            sell_vol += qty
        else:
            buy_vol += qty
    total = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / total * 100 if total else 0.0
    return round(imbalance,2), round(buy_vol,2), round(sell_vol,2)

# === Enhanced data fetching ===
@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    """Fetch enhanced data with technical indicators, VWAP, and order flow using Binance Futures API"""
    attempts = []
    try:
        # Fetch 24hr stats from Futures endpoint
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        stats = response.json()

        # Fetch kline data
        kline_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit=100"
        kline_response = requests.get(kline_url, timeout=5)
        kline_response.raise_for_status()
        klines = kline_response.json()

        # Fetch recent trades for order flow
        trades_url = f"https://fapi.binance.com/fapi/v1/trades?symbol={symbol}&limit=500"
        trades_response = requests.get(trades_url, timeout=5)
        trades_response.raise_for_status()
        trades = trades_response.json()

        # Compute indicators
        closes = [float(k[4]) for k in klines]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-14:].mean()
        avg_loss = losses[-14:].mean() if losses[-14:].mean() > 0 else 1
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        if len(closes) >= 26:
            ema12 = np.mean(closes[-12:])
            ema26 = np.mean(closes[-26:])
            macd = (ema12 - ema26) / ema26 * 100
        else:
            macd = 0

        # VWAP and bands
        vwap_calc, vwap_up, vwap_low = calculate_vwap(klines)
        vwap = vwap_calc if not np.isnan(vwap_calc) else float(stats['lastPrice'])
        vwap_upper = vwap_up
        vwap_lower = vwap_low

        # Order flow imbalance
        imbalance, buy_vol, sell_vol = calculate_order_flow_imbalance(trades)

        return {
            'data_source': 'Binance Futures',
            'price': float(stats['lastPrice']),
            'price_change_24h': float(stats['priceChangePercent']),
            'volume': float(stats['volume']),
            'high_24h': float(stats['highPrice']),
            'low_24h': float(stats['lowPrice']),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'vwap': round(vwap, 2),
            'vwap_upper': round(vwap_upper, 2),
            'vwap_lower': round(vwap_lower, 2),
            'price_vs_vwap': round((float(stats['lastPrice']) - vwap) / vwap * 100, 2),
            'order_flow_imbalance': imbalance,
            'aggressive_buy_volume': buy_vol,
            'aggressive_sell_volume': sell_vol,
            'last_updated': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        attempts.append(f"Futures API error: {str(e)[:50]}")

    # Fallback to demo mode
    return {
        'data_source': f'Demo Mode ({len(attempts)} failures)',
        'price': np.nan,
        'price_change_24h': np.nan,
        'volume': np.nan,
        'high_24h': np.nan,
        'low_24h': np.nan,
        'rsi': np.nan,
        'macd': np.nan,
        'vwap': np.nan,
        'vwap_upper': np.nan,
        'vwap_lower': np.nan,
        'price_vs_vwap': np.nan,
        'order_flow_imbalance': np.nan,
        'aggressive_buy_volume': 0,
        'aggressive_sell_volume': 0,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    }

# Signal probability calculation
# You can customize this further

def calculate_signal_probability(data):
    base_prob = 50
    if not np.isnan(data['price']) and not np.isnan(data['vwap']):
        base_prob += (data['price'] > data['vwap']) * 10
    # RSI adjustments
    if data['rsi'] > 70:
        base_prob -= 10
    elif data['rsi'] < 30:
        base_prob += 10
    # MACD adjustments
    base_prob += np.sign(data['macd']) * 5
    return base_prob

# Generate trading signals based on strategy

def generate_trading_signals(data, strategy):
    signals = {}
    price = data['price']
    if np.isnan(price):
        signals[strategy] = 'no_data'
        return signals
    if strategy == 'day_trading':
        signals[strategy] = 'buy' if price > data['vwap'] else 'sell'
    elif strategy == 'perp':
        signals[strategy] = 'long' if price > data['vwap'] else 'short'
    elif strategy == 'spot':
        signals[strategy] = 'buy' if data['rsi'] < 30 else 'hold'
    else:
        # All strategies
        signals = {
            'day_trading': 'buy' if price > data['vwap'] else 'sell',
            'perp': 'long' if price > data['vwap'] else 'short',
            'spot': 'buy' if data['rsi'] < 30 else 'hold'
        }
    return signals

# Save signals to history

def save_signal_to_history(symbol, strategy, signal):
    st.session_state.signal_history.append({
        'symbol': symbol,
        'strategy': strategy,
        'signal': signal,
        'timestamp': datetime.now()
    })

# === Main rendering ===

col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
with col_status1:
    selected_strategy = st.radio(
        "Trading Strategy:",
        ["Perpetual (Futures)", "Day Trading", "Spot Trading", "All Strategies"],
        horizontal=True
    )
with col_status2:
    if symbols:
        test_data = get_enhanced_data(symbols[0])
        status = "🟢 Live" if test_data['data_source'].startswith('Binance Futures') else "🟡 Demo"
        st.metric("Status", status)
with col_status3:
    st.metric("Last Update", st.session_state.last_update.strftime('%H:%M:%S'))

# Display metrics and charts per symbol
cols = st.columns(len(symbols))
strategy_key = {
    "Perpetual (Futures)": 'perp',
    "Day Trading": 'day_trading',
    "Spot Trading": 'spot',
    "All Strategies": 'all'
}
for i, sym in enumerate(symbols):
    with cols[i]:
        data = get_enhanced_data(sym)
        prob = calculate_signal_probability(data)
        st.subheader(f"{sym} Signal Probability")
        st.metric(label="Probability", value=f"{prob}%")

        # Generate and save signals
        sigs = generate_trading_signals(data, strategy_key[selected_strategy])
        sig = sigs[strategy_key[selected_strategy]]
        save_signal_to_history(sym, strategy_key[selected_strategy], sig)
        st.write(f"Signal: {sig}")

        # Price & VWAP chart
        df = pd.DataFrame([{
            'time': datetime.now(),
            'price': data['price'],
            'vwap': data['vwap']
        }])
        fig = make_subplots(specs=[[{'secondary_y': False}]])
        fig.add_trace(go.Scatter(x=df['time'], y=df['price'], name='Price'))
        fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], name='VWAP', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

# Strategy guidelines
st.markdown("""
### Trading Guidelines

- Only take longs with positive flow
- Only take shorts with negative flow
- Avoid trades against strong flow

### Risk Management with VWAP
- Use VWAP bands as stop loss levels
- Reduce position size when extended from VWAP
- Take partial profits at VWAP
""")

# Auto-refresh logic
current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
