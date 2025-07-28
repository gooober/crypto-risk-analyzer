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

# Debug store
if 'debug_log' not in st.session_state:
    st.session_state.debug_log = []

def log_debug(msg):
    st.session_state.debug_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

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

    st.header("🔔 Alert Settings")
    enable_alerts = st.checkbox("Enable Alerts", value=False)
    if enable_alerts:
        alert_prob_threshold = st.slider("Alert when probability >", 60, 90, 70)
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

    st.header("🛠️ Debug & Status")
    if st.session_state.get('debug_log'):
        st.info('\n'.join(st.session_state.debug_log[-5:]))

# Helper Functions

def safe_api_get(url):
    """Fetch and check a Binance endpoint, log errors."""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            log_debug(f"HTTP {resp.status_code} for {url}")
            return None
        j = resp.json()
        if not j:
            log_debug(f"No JSON data for {url}")
            return None
        return j
    except Exception as e:
        log_debug(f"Exception for {url}: {str(e)[:60]}")
        return None

def calculate_vwap(klines):
    try:
        volumes = [float(k[5]) for k in klines]
        typicals = [ (float(k[2]) + float(k[3]) + float(k[4]))/3 for k in klines ]
        tp_vol = [t*v for t,v in zip(typicals, volumes)]
        cum_tp_vol = np.sum(tp_vol)
        cum_vol = np.sum(volumes)
        vwap = cum_tp_vol / cum_vol if cum_vol else np.nan
        assert np.isfinite(vwap)
        return vwap, vwap*1.002, vwap*0.998
    except Exception as e:
        log_debug(f"VWAP calc error: {e}")
        return np.nan, np.nan, np.nan

def calculate_order_flow_imbalance(trades):
    buy_vol, sell_vol = 0.0, 0.0
    try:
        for trade in trades:
            qty = float(trade.get('qty', trade.get('quantity', 0)))
            if trade.get('isBuyerMaker', False): sell_vol += qty
            else: buy_vol += qty
        total = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total * 100 if total else 0.0
        assert np.isfinite(imbalance)
        return round(imbalance,2), round(buy_vol,2), round(sell_vol,2)
    except Exception as e:
        log_debug(f"Order flow calc error: {e}")
        return 0.0, 0.0, 0.0

@st.cache_data(ttl=5)
def get_enhanced_data(symbol):
    # Pull data, check after each step
    stats = safe_api_get(f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}")
    if not stats:
        return {'data_source': 'Demo: 24hr fail', 'price': np.nan, 'vwap': np.nan, 'rsi': np.nan, 'macd': np.nan}
    klines = safe_api_get(f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit=100")
    if not klines:
        return {'data_source': 'Demo: klines fail', 'price': np.nan, 'vwap': np.nan, 'rsi': np.nan, 'macd': np.nan}
    trades = safe_api_get(f"https://fapi.binance.com/fapi/v1/trades?symbol={symbol}&limit=500")
    if not trades:
        return {'data_source': 'Demo: trades fail', 'price': np.nan, 'vwap': np.nan, 'rsi': np.nan, 'macd': np.nan}

    closes = [float(k[4]) for k in klines]
    try:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains[-14:].mean()
        avg_loss = losses[-14:].mean() if losses[-14:].mean() > 0 else 1
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        assert np.isfinite(rsi)
    except Exception as e:
        log_debug(f"RSI calc error: {e}")
        rsi = np.nan

    try:
        if len(closes) >= 26:
            ema12 = np.mean(closes[-12:])
            ema26 = np.mean(closes[-26:])
            macd = (ema12 - ema26) / ema26 * 100
        else:
            macd = 0
        assert np.isfinite(macd)
    except Exception as e:
        log_debug(f"MACD calc error: {e}")
        macd = np.nan

    vwap, vwap_up, vwap_low = calculate_vwap(klines)
    imbalance, buy_vol, sell_vol = calculate_order_flow_imbalance(trades)
    try:
        price = float(stats['lastPrice'])
        assert np.isfinite(price)
    except Exception as e:
        log_debug(f"Price parse error: {e}")
        price = np.nan

    # Final check
    if np.isnan(price) or np.isnan(vwap):
        return {'data_source': 'Demo: nan in price/vwap', 'price': np.nan, 'vwap': np.nan, 'rsi': np.nan, 'macd': np.nan}

    return {
        'data_source': 'Binance Futures',
        'price': price,
        'rsi': round(rsi, 2),
        'macd': round(macd, 2),
        'vwap': round(vwap, 2),
        'vwap_upper': round(vwap_up, 2),
        'vwap_lower': round(vwap_low, 2),
        'order_flow_imbalance': imbalance,
        'aggressive_buy_volume': buy_vol,
        'aggressive_sell_volume': sell_vol,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    }

def calculate_signal_probability(data):
    base_prob = 50
    if not np.isnan(data['price']) and not np.isnan(data['vwap']):
        base_prob += (data['price'] > data['vwap']) * 10
    if data['rsi'] > 70: base_prob -= 10
    elif data['rsi'] < 30: base_prob += 10
    base_prob += np.sign(data['macd']) * 5
    return base_prob

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
        signals = {
            'day_trading': 'buy' if price > data['vwap'] else 'sell',
            'perp': 'long' if price > data['vwap'] else 'short',
            'spot': 'buy' if data['rsi'] < 30 else 'hold'
        }
    return signals

def save_signal_to_history(symbol, strategy, signal):
    st.session_state.signal_history.append({
        'symbol': symbol,
        'strategy': strategy,
        'signal': signal,
        'timestamp': datetime.now()
    })

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
        status = "🟢 Live" if test_data['data_source'] == 'Binance Futures' else f"🟡 Demo ({test_data['data_source']})"
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
