import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Intraday Crypto Trade Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Intraday Crypto Trade Risk Analyzer")

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

    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        st.slider("Alert when probability >", 60, 90, 70)

    if st.button("🗑️ Clear Alerts"):
        st.session_state.alerts = []

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

def calculate_vwap(klines):
    volumes = [float(k[5]) for k in klines]
    typicals = [ (float(k[2]) + float(k[3]) + float(k[4]))/3 for k in klines ]
    tp_vol = [t*v for t,v in zip(typicals, volumes)]
    cum_tp_vol = np.sum(tp_vol)
    cum_vol = np.sum(volumes)
    vwap = cum_tp_vol / cum_vol if cum_vol else np.nan
    return vwap, vwap * 1.002, vwap * 0.998

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
    # 1. Try Bybit Perpetuals API
    try:
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['result']['list'][0]

        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['result']['list']
        closes = [float(k[4]) for k in klines]
        vwap, vwap_up, vwap_low = calculate_vwap(klines)
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
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
            'vwap_upper': round(vwap_up, 2),
            'vwap_lower': round(vwap_low, 2),
            'price_vs_vwap': round((float(ticker_data['lastPrice']) - vwap) / vwap * 100, 2),
            'order_flow_imbalance': np.nan,
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'last_updated': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e1:
        st.sidebar.warning(f"Bybit API failed: {str(e1)[:60]}")

    # 2. Try OKX Spot API
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['data'][0]

        kline_url = f"https://www.okx.com/api/v5/market/candles?instId={okx_symbol}&bar=1m&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['data']
        # OKX candle: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        closes = [float(k[4]) for k in klines]
        vwap, vwap_up, vwap_low = calculate_vwap([[None,None,float(k[2]),float(k[3]),float(k[4]),float(k[5])] for k in klines])
        rsi = calculate_rsi(closes)
        macd = calculate_macd(closes)
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
            'vwap_upper': round(vwap_up, 2),
            'vwap_lower': round(vwap_low, 2),
            'price_vs_vwap': round((float(ticker_data['last']) - vwap) / vwap * 100, 2) if vwap else np.nan,
            'order_flow_imbalance': np.nan,
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'last_updated': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e2:
        st.sidebar.warning(f"OKX API failed: {str(e2)[:60]}")

    # 3. Try Kraken Spot API (price only)
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
            'vwap_upper': np.nan,
            'vwap_lower': np.nan,
            'price_vs_vwap': np.nan,
            'order_flow_imbalance': np.nan,
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'last_updated': datetime.now().strftime('%H:%M:%S')
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
        'vwap_upper': np.nan,
        'vwap_lower': np.nan,
        'price_vs_vwap': np.nan,
        'order_flow_imbalance': np.nan,
        'aggressive_buy_volume': 0,
        'aggressive_sell_volume': 0,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    }

def calculate_signal_probability(data):
    base_prob = 50
    if not np.isnan(data['price']) and not np.isnan(data['vwap']):
        base_prob += (data['price'] > data['vwap']) * 10
    if not np.isnan(data['rsi']):
        if data['rsi'] > 70:
            base_prob -= 10
        elif data['rsi'] < 30:
            base_prob += 10
    if not np.isnan(data['macd']):
        base_prob += np.sign(data['macd']) * 5
    return base_prob

def generate_trading_signals(data, strategy):
    signals = {}
    price = data['price']
    if np.isnan(price):
        signals[strategy] = 'no_data'
        return signals
    if strategy == 'day_trading':
        signals[strategy] = 'buy' if not np.isnan(data['vwap']) and price > data['vwap'] else 'sell'
    elif strategy == 'perp':
        signals[strategy] = 'long' if not np.isnan(data['vwap']) and price > data['vwap'] else 'short'
    elif strategy == 'spot':
        signals[strategy] = 'buy' if not np.isnan(data['rsi']) and data['rsi'] < 30 else 'hold'
    else:
        signals = {
            'day_trading': 'buy' if not np.isnan(data['vwap']) and price > data['vwap'] else 'sell',
            'perp': 'long' if not np.isnan(data['vwap']) and price > data['vwap'] else 'short',
            'spot': 'buy' if not np.isnan(data['rsi']) and data['rsi'] < 30 else 'hold'
        }
    return signals

def save_signal_to_history(symbol, strategy, signal):
    st.session_state.signal_history.append({
        'symbol': symbol,
        'strategy': strategy,
        'signal': signal,
        'timestamp': datetime.now()
    })

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

        sigs = generate_trading_signals(data, strategy_key[selected_strategy])
        sig = sigs[strategy_key[selected_strategy]]
        save_signal_to_history(sym, strategy_key[selected_strategy], sig)
        st.write(f"Signal: {sig}")

        # Chart
        df = pd.DataFrame([{
            'time': datetime.now(),
            'price': data['price'],
            'vwap': data['vwap']
        }])
        fig = make_subplots(specs=[[{'secondary_y': False}]])
        fig.add_trace(go.Scatter(x=df['time'], y=df['price'], name='Price'))
        if not np.isnan(data['vwap']):
            fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], name='VWAP', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

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

current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
