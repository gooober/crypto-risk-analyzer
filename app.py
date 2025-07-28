import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ========== AI Integration ==========
def ai_trade_commentary(symbol, price, vwap, supertrend, signal, stop, balance, leverage, position_size):
    try:
        import openai
        openai.api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"You are a trading assistant for a day trader with ${balance} and {leverage}x leverage. "
            f"They want to risk only ${position_size} on this trade. "
            f"Symbol: {symbol}. Price: {price:.2f}, VWAP: {vwap:.2f}, SuperTrend: {supertrend:.2f}. "
            f"Signal: {signal}. Stop Loss: {stop:.2f}. "
            f"Give clear advice on 1) if this is a smart trade for this account size, "
            f"2) why or why not to enter now, 3) any warnings about risk, 4) a one-sentence summary."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=120
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI commentary not available ({e})"

# ========== Streamlit UI ==========
st.set_page_config(
    page_title="Small Account Day Trading Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚦 Small Account Day Trading Analyzer")

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# --- Sidebar (global account settings and API key) ---
with st.sidebar:
    st.header("Global Account & Risk Settings")
    default_balance = st.number_input("Default Account Size (USD)", min_value=10.0, value=50.0, step=1.0)
    max_risk_per_trade = st.number_input("Max Risk % per Trade", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    leverage = st.slider("Leverage", 1, 20, 5)
    ai_enabled = st.checkbox("Enable AI commentary (OpenAI key)", value=False)
    if ai_enabled:
        openai_key = st.text_input("Paste your OpenAI API key", type="password")
        if openai_key:
            st.session_state["openai_api_key"] = openai_key
    st.header("Symbols & Refresh")
    symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        default=["BTCUSDT"]
    )
    refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 10)
    if st.button("🔄 Refresh Now"):
        st.session_state.last_update = datetime.now()
        st.rerun()

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
    hl2 = (df['high'] + df['low']) / 2
    atr = pd.Series(df['high'] - df['low']).rolling(period).mean()
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    supertrend = [np.nan]*len(df)
    direction = [1]
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

# --- Universal Data Fetcher with US-friendly fallback ---
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
    try:
        # Bybit
        url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['result']['list'][0]
        kline_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['result']['list']
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        vwap = calculate_vwap(klines)
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows})
        supertrend, direction = calculate_supertrend(df)
        return {
            'data_source': 'Bybit Perpetuals',
            'price': float(ticker_data['lastPrice']),
            'vwap': round(vwap, 2),
            'supertrend': supertrend,
            'supertrend_dir': direction,
            'signal': "Buy" if direction[-1] == 1 and closes[-1] > supertrend[-1] else "Sell",
            'stop_loss': round(supertrend[-1], 2),
            'price_series': closes
        }
    except Exception as e1:
        st.sidebar.warning(f"Bybit API failed: {str(e1)[:60]}")
    try:
        # OKX Spot
        url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}"
        resp = requests.get(url, timeout=5)
        ticker_data = resp.json()['data'][0]
        kline_url = f"https://www.okx.com/api/v5/market/candles?instId={okx_symbol}&bar=1m&limit=100"
        klines = requests.get(kline_url, timeout=5).json()['data']
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        vwap = calculate_vwap([[None, None, float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in klines])
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows})
        supertrend, direction = calculate_supertrend(df)
        return {
            'data_source': 'OKX Spot',
            'price': float(ticker_data['last']),
            'vwap': round(vwap, 2),
            'supertrend': supertrend,
            'supertrend_dir': direction,
            'signal': "Buy" if direction[-1] == 1 and closes[-1] > supertrend[-1] else "Sell",
            'stop_loss': round(supertrend[-1], 2),
            'price_series': closes
        }
    except Exception as e2:
        st.sidebar.warning(f"OKX API failed: {str(e2)[:60]}")
    try:
        # Kraken
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
        resp = requests.get(url, timeout=5)
        result = resp.json()['result']
        first_key = list(result.keys())[0]
        ticker_data = result[first_key]
        price = float(ticker_data['c'][0])
        return {
            'data_source': 'Kraken Spot',
            'price': price,
            'vwap': np.nan,
            'supertrend': [np.nan],
            'supertrend_dir': [0],
            'signal': "No Signal",
            'stop_loss': np.nan,
            'price_series': [price]
        }
    except Exception as e3:
        st.sidebar.warning(f"Kraken API failed: {str(e3)[:60]}")
    st.sidebar.error("All live APIs failed. Showing demo/offline mode.")
    return {
        'data_source': f'Demo Mode (All APIs failed)',
        'price': np.nan,
        'vwap': np.nan,
        'supertrend': [np.nan],
        'supertrend_dir': [0],
        'signal': "No Signal",
        'stop_loss': np.nan,
        'price_series': []
    }

# --- TABS ---
tabs = st.tabs(["Dashboard", "Trade Planner"])

with tabs[0]:
    st.markdown("### Dashboard (All Symbols)")
    for sym in symbols:
        data = get_enhanced_data(sym)
        st.subheader(f"{sym}")
        st.metric("Price", f"${data['price']:.2f}" if not np.isnan(data['price']) else "N/A")
        st.metric("VWAP", f"{data['vwap']:.2f}" if not np.isnan(data['vwap']) else "N/A")
        st.metric("SuperTrend", f"{data['supertrend'][-1]:.2f}" if not np.isnan(data['supertrend'][-1]) else "N/A")
        st.metric("Signal", data['signal'])
        st.metric("Stop Loss", f"${data['stop_loss']:.2f}" if not np.isnan(data['stop_loss']) else "N/A")

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

        st.session_state.signal_history.append({
            'symbol': sym,
            'signal': data['signal'],
            'price': data['price'],
            'timestamp': datetime.now()
        })

with tabs[1]:
    st.markdown("### Trade Planner")
    for sym in symbols:
        data = get_enhanced_data(sym)
        st.subheader(f"{sym} Trade Planner")

        buy_in = st.number_input(f"Your Buy-in Amount for {sym} ($)", min_value=1.0, value=default_balance, step=1.0, key=f"buyin_{sym}")
        entry_price = data['price']
        stop = data['stop_loss']

        # --- Position sizing logic ---
        risk_dollars = buy_in * max_risk_per_trade / 100
        if not np.isnan(entry_price) and not np.isnan(stop) and abs(entry_price - stop) > 0:
            pos_size = round(risk_dollars / abs(entry_price - stop) * entry_price / leverage, 2)
            st.info(f"**Suggested Position Size:** ${pos_size} (at {leverage}x, risking {max_risk_per_trade}% of your buy-in)")
            trade_advice = (
                f"{'🟢 **BUY NOW**' if data['signal']=='Buy' else '🔴 **SELL/AVOID**'} at ${entry_price:.2f}, "
                f"Stop Loss: ${stop:.2f}. (Risk: ${risk_dollars:.2f})"
            )
            st.success(trade_advice)
        else:
            st.info("Position size not available (price or stop missing).")

        # --- AI Commentary (optional) ---
        if ai_enabled and not np.isnan(entry_price) and not np.isnan(stop) and not np.isnan(data['vwap']) and not np.isnan(data['supertrend'][-1]):
            ai_result = ai_trade_commentary(
                sym, entry_price, data['vwap'], data['supertrend'][-1], data['signal'], stop,
                buy_in, leverage, pos_size if 'pos_size' in locals() else 0
            )
            st.warning("**AI Commentary:**\n" + ai_result)

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

st.markdown("""
### How to Use Trade Planner
- Enter your buy-in for each symbol in the Trade Planner tab.
- You’ll get a clear “Buy Now” or “Sell/Avoid” signal, stop loss, and personalized position size.
- AI commentary (if enabled) will check each setup for extra risk or warnings.
""")

current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= refresh_rate:
    st.session_state.last_update = current_time
    st.rerun()
