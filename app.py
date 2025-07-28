import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Perpetual Crypto Trade Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Perpetual Crypto Trade Risk Analyzer")

# Session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# --- Sidebar ---
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
    st.header("🔔 Alerts")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        st.slider("Alert when probability >", 60, 90, 70)
    if st.button("📜 View Signal History"):
        history = st.session_state.signal_history
        if history:
            st.write(f"Last {min(5, len(history))} signals:")
            for sig in history[-5:]:
                st.write(f"• {sig['timestamp'].strftime('%H:%M')} - {sig['symbol']} {sig['signal']} @ {sig['price']}")

# --- Indicator helpers ---
def calculate_vwap(klines):
    volumes = [float(k[5]) for k in klines]
    typicals = [(float(k[2]) + float(k[3]) + float(k[4]))/3 for k in klines]
    tp_vol = [t*v for t, v in zip(typicals, volumes)]
    cum_tp_vol = np.sum(tp_vol)
    cum_vol = np.sum(volumes)
    vwap = cum_tp_vol / cum_vol if cum_vol else np.nan
    return vwap

def calculate_supertrend(df, period=10, multiplier=3):
    """Returns SuperTrend line, direction (1=long, -1=short), buy/sell points, stop"""
    hl2 = (df['high'] + df['low']) / 2
    atr = pd.Series(df['high'] - df['low']).rolling(period).mean()
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    supertrend = [np.nan]*len(df)
    direction = [1]  # 1=long, -1=short
    for i in range(1, len(df)):
        if df['close'][i] > final_upperband[i-1]:
            direction.append(1)
        elif df['close'][i] < final_lowerband[i-1]:
            direction.append(-1)
        else:
            direction.append(direction[-1])
        if direction[-1] == 1:
            supertrend[i] = max(final_lowerband[i], supertrend[i-1] if i > 0 and direction[i-1] == 1 else final_lowerband[i])
        else:
            supertrend[i] = min(final_upperband[i], supertrend[i-1] if i > 0 and direction[i-1] == -1 else final_upperband[i])
    supertrend = pd.Series(supertrend).fillna(method='bfill').to_list()
    return supertrend, direction

def calculate_rsi(closes, period=14):
    closes = np.array(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean() if losses[-period:].mean() > 0 else 1
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(closes, fast=12, slow=26):
    closes = np.array(closes)
    if len(closes) < slow:
        return np.nan
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().iloc[-1]
    return (ema_fast - ema_slow) / ema_slow * 100

# --- Universal Data Fetcher ---
def get_enhanced_data(symbol):
    okx_symbol = symbol.replace("USDT", "-USDT")
    kraken_map = {
        "BTCUSDT": "XBTUSDT",
        "ETHUSDT": "ETHUSDT",
        "BNBUSDT": "BNBUSDT",
        "SOLUSDT": "SOLUSDT",
        "ADAUSDT": "ADAUSDT"
    }
    kraken_symbol = kraken_map.get(symbol, symbol)
    # 1. Bybit
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['result']['list'][0]
        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['result']['list']
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        vwap = calculate_vwap(klines)
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        df = pd.DataFrame({
            "close": closes,
            "high": highs,
            "low": lows
        })
        supertrend, direction = calculate_supertrend(df)
        return {
            'data_source': 'Bybit Perpetuals',
            'price': float(ticker_data['lastPrice']),
            'price_change_24h': float(ticker_data['price24hPcnt']) * 100,
            'volume': float(ticker_data['turnover24h']),
            'high_24h': float(ticker_data['highPrice24h']),
            'low_24h': float(ticker_data['lowPrice24h']),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'vwap': round(vwap, 2),
            'supertrend': supertrend,
            'supertrend_dir': direction,
            'signal': "Buy" if direction[-1] == 1 and closes[-1] > supertrend[-1] else "Sell",
            'stop_loss': round(supertrend[-1], 2),
            'last_updated': datetime.now().strftime('%H:%M:%S'),
            "price_series": closes
        }
    except Exception as e1:
        st.sidebar.warning(f"Bybit API failed: {str(e1)[:60]}")

    # 2. OKX Spot
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['data'][0]
        kline_url = f"https://www.okx.com/api/v5/market/candles?instId={okx_symbol}&bar=1m&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['data']
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        vwap = calculate_vwap([[None, None, float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines])
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
        df = pd.DataFrame({
            "close": closes,
            "high": highs,
            "low": lows
        })
        supertrend, direction = calculate_supertrend(df)
        return {
            'data_source': 'OKX Spot',
            'price': float(ticker_data['last']),
            'price_change_24h': float(ticker_data.get('change24h', 0)),
            'volume': float(ticker_data.get('volCcy24h', 0)),
            'high_24h': float(ticker_data.get('high24h', 0)),
            'low_24h': float(ticker_data.get('low24h', 0)),
            'rsi': round(rsi, 2),
            'macd': round(macd, 2),
            'vwap': round(vwap, 2),
            'supertrend': supertrend,
            'supertrend_dir': direction,
            'signal': "Buy" if direction[-1] == 1 and closes[-1] > supertrend[-1] else "Sell",
            'stop_loss': round(supertrend[-1], 2),
            'last_updated': datetime.now().strftime('%H:%M:%S'),
            "price_series": closes
        }
    except Exception as e2:
        st.sidebar.warning(f"OKX API failed: {str(e2)[:60]}")

    # 3. Kraken
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
        resp = requests.get(url, timeout=5)
        result = resp.json()['result']
        first_key = list(result.keys())[0]
        ticker_data = result[first_key]
        price = float(ticker_data['c'][0])
        return {
            'data_source': 'Kraken Spot',
            'price': price,
            'price_change_24h': np.nan,
            'volume': float(ticker_data['v'][1]),
            'high_24h': float(ticker_data['h'][1]),
            'low_24h': float(ticker_data['l'][1]),
            'rsi': np.nan,
            'macd': np.nan,
            'vwap': np.nan,
            'supertrend': [np.nan],
            'supertrend_dir': [0],
            'signal': "No Signal",
            'stop_loss': np.nan,
            'last_updated': datetime.now().strftime('%H:%M:%S'),
            "price_series": [price]
        }
    except Exception as e3:
        st.sidebar.warning(f"Kraken API failed: {str(e3)[:60]}")

    st.sidebar.error("All live APIs failed. Showing demo/offline mode.")
    return {
        'data_source': f'Demo Mode (All APIs failed)',
        'price': np.nan,
        'price_change_24h': np.nan,
        'volume': np.nan,
        'high_24h': np.nan,
        'low_24h': np.nan,
        'rsi': np.nan,
        'macd': np.nan,
        'vwap': np.nan,
        'supertrend': [np.nan],
        'supertrend_dir': [0],
        'signal': "No Signal",
        'stop_loss': np.nan,
        'last_updated': datetime.now().strftime('%H:%M:%S'),
        "price_series": []
    }

# --- Main Panel ---
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
        status = "🟢 Live" if "Demo" not in test_data['data_source'] else "🟡 Demo"
        st.metric("Status", status)
with col_status3:
    st.metric("Last Update", st.session_state.last_update.strftime('%H:%M:%S'))

cols = st.columns(len(symbols))
for i, sym in enumerate(symbols):
    with cols[i]:
        data = get_enhanced_data(sym)
        st.subheader(f"{sym}")
        st.metric("Price", f"${data['price']:.2f}" if not np.isnan(data['price']) else "N/A")
        st.metric("VWAP", f"{data['vwap']:.2f}" if not np.isnan(data['vwap']) else "N/A")
        st.metric("SuperTrend", f"{data['supertrend'][-1]:.2f}" if not np.isnan(data['supertrend'][-1]) else "N/A")
        st.metric("Signal", data['signal'])
        st.metric("Stop Loss", f"${data['stop_loss']:.2f}" if not np.isnan(data['stop_loss']) else "N/A")

        st.markdown(
            f"""
            **Buy:** { 'YES' if data['signal'] == 'Buy' else 'NO' }  
            **Sell:** { 'YES' if data['signal'] == 'Sell' else 'NO' }  
            **Suggested Stop:** {data['stop_loss'] if not np.isnan(data['stop_loss']) else "N/A"}
            """
        )

        # --- Chart: Price, VWAP, SuperTrend ---
        if data['price_series'] and not np.isnan(data['vwap']):
            price_series = data['price_series']
            x = [datetime.now() - pd.Timedelta(minutes=len(price_series)-i-1) for i in range(len(price_series))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=price_series, name="Price"))
            fig.add_trace(go.Scatter(x=x, y=[data['vwap']]*len(price_series), name="VWAP", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=data['supertrend'], name="SuperTrend", line=dict(color='green' if data['signal']=='Buy' else 'red')))
            fig.update_layout(margin=dict(t=10,b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)

        # --- Signal History Save ---
        st.session_state.signal_history.append({
            'symbol': sym,
            'signal': data['signal'],
            'price': data['price'],
            'timestamp': datetime.now()
        })

st.markdown("""
### How This Works

- **Buy:** When price is above SuperTrend and SuperTrend is green.
- **Sell:** When price is below SuperTrend and SuperTrend is red.
- **Suggested Stop Loss:** Last SuperTrend line (auto-adjusts every minute).
- VWAP gives fair value; price crossing above/below is a confirmation.

**All data is live (Bybit/OKX/Kraken fallback) and auto-refreshes every few seconds.**
""")

current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
